"""Back up and permanently complete the canonical synthetic v3 sensor files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GRID_DIR = DATA / "dr_dataset_v3_grid_et_code"
BACKUP = DATA / "original_v3_backup"
GRID = GRID_DIR / "grid_1min.csv"
HOUSEHOLDS = [DATA / "households_1min_C001_C025.parquet", DATA / "households_1min_C026_C050.parquet"]
INFO = GRID_DIR / "households_info.csv"
GRID_SENSORS = ["temperature_c", "households_consumption_kw", "steg_production_kw", "pv_production_kw"]
HOUSEHOLD_SENSORS = ["aggregate_power_w", "ac_power_w", "water_heater_power_w", "washing_machine_power_w"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def ensure_backups() -> tuple[Path, list[Path], dict[str, str]]:
    BACKUP.mkdir(parents=True, exist_ok=True)
    sources = [GRID, *HOUSEHOLDS]
    targets = [BACKUP / p.name for p in sources]
    record_path = BACKUP / "backup_checksums.json"
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        for target in targets:
            if not target.is_file() or sha256(target) != record[target.name]:
                raise AssertionError(f"Backup checksum mismatch: {target}")
        return targets[0], targets[1:], record
    record = {}
    for source, target in zip(sources, targets):
        temp = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(source, temp)
        if sha256(source) != sha256(temp): raise AssertionError(f"Backup copy mismatch: {source}")
        os.replace(temp, target); record[target.name] = sha256(target)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return targets[0], targets[1:], record


def complete(values: pd.Series, timestamps: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    original = numeric.copy(); missing = original.isna()
    indexed = pd.Series(original.to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(timestamps)))
    internal = indexed.interpolate(method="time", limit_area="inside")
    internal_count = int((missing & pd.Series(internal.to_numpy(), index=values.index).notna()).sum())
    # Deterministic boundary policy: nearest valid observation, used only if an edge remains.
    final = internal.ffill().bfill()
    result = pd.Series(final.to_numpy(), index=values.index).clip(lower=0)
    boundary_count = int(missing.sum()) - internal_count
    if not original.loc[~missing].equals(result.loc[~missing]): raise AssertionError("Valid sensor value changed")
    return result, {"original_missing":int(missing.sum()), "linear_interpolated":internal_count,
                    "boundary_nearest_value":boundary_count, "remaining_missing":int(result.isna().sum())}


def complete_grid(source: Path, target: Path) -> tuple[Path, dict[str, object]]:
    frame = pd.read_csv(source, parse_dates=["timestamp"], low_memory=False)
    original_labels = frame[["is_holiday","is_ramadan","is_dr_event","is_dr_peak"]].copy()
    report = {}
    for channel in GRID_SENSORS:
        frame[channel], report[channel] = complete(frame[channel], frame.timestamp)
    missing_peak = frame.is_dr_peak.isna()
    reconstructed = (frame.households_consumption_kw.rolling(15, center=True, min_periods=1).mean()
                     > frame.steg_production_kw + frame.pv_production_kw).astype("int8")
    frame.loc[missing_peak,"is_dr_peak"] = reconstructed.loc[missing_peak]
    frame["is_dr_peak"] = frame.is_dr_peak.astype("int8")
    for column in ["is_holiday","is_ramadan","is_dr_event"]:
        if not frame[column].equals(original_labels[column]): raise AssertionError(f"Label changed: {column}")
    if not frame.loc[~missing_peak,"is_dr_peak"].astype(float).equals(original_labels.loc[~missing_peak,"is_dr_peak"]):
        raise AssertionError("Existing peak labels changed")
    if frame.timestamp.duplicated().any() or frame[GRID_SENSORS].isna().any().any(): raise AssertionError("Grid validation failed")
    temp = target.with_suffix(target.suffix + ".complete.tmp")
    frame.to_csv(temp,index=False,date_format="%Y-%m-%d %H:%M")
    report["is_dr_peak"] = {"deterministically_reconstructed":int(missing_peak.sum()),"rule":"centered 15-minute mean feeder demand > STEG + PV"}
    return temp, report


def complete_households(source: Path, target: Path, info: pd.DataFrame) -> tuple[Path, dict[str, object]]:
    first_half = "C001_C025" in source.name
    clients = info.loc[info.client_id.between("C001","C025") if first_half else info.client_id.between("C026","C050"),"client_id"].tolist()
    submeter = info.set_index("client_id").has_submeter.to_dict()
    report = {c:{"original_missing":0,"linear_interpolated":0,"boundary_nearest_value":0,"remaining_missing":0,"structurally_unavailable":0} for c in HOUSEHOLD_SENSORS}
    temp = target.with_suffix(target.suffix + ".complete.tmp"); writer=None
    try:
        for client in clients:
            frame=pd.read_parquet(source,filters=[("client_id","==",client)],engine="pyarrow").sort_values("timestamp",kind="stable").reset_index(drop=True)
            if len(frame)!=525_600 or frame.timestamp.duplicated().any(): raise AssertionError(f"Timeline invalid: {client}")
            # Stable schema across panel/non-panel homes; structural nulls remain null floats.
            for channel in HOUSEHOLD_SENSORS: frame[channel]=pd.to_numeric(frame[channel],errors="coerce").astype("float64")
            for channel in HOUSEHOLD_SENSORS:
                installed = channel=="aggregate_power_w" or bool(submeter[client])
                if installed:
                    frame[channel],stats=complete(frame[channel],frame.timestamp)
                    for key,value in stats.items(): report[channel][key]+=value
                else:
                    report[channel]["structurally_unavailable"]+=int(frame[channel].isna().sum())
                if installed and (frame[channel].isna().any() or (frame[channel]<0).any()):
                    raise AssertionError(f"Invalid completed sensor: {client}/{channel}")
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None: writer=pq.ParquetWriter(temp,table.schema,compression="zstd",compression_level=9,use_dictionary=["client_id"])
            writer.write_table(table,row_group_size=131_400)
    finally:
        if writer is not None: writer.close()
    return temp, report


def main() -> None:
    backup_grid, backup_households, backup_hashes = ensure_backups()
    info=pd.read_csv(INFO)
    grid_temp,grid_report=complete_grid(backup_grid,GRID)
    household_outputs=[]; household_report={}
    for source,target in zip(backup_households,HOUSEHOLDS):
        temp,report=complete_households(source,target,info); household_outputs.append((temp,target)); household_report[target.name]=report
    # All outputs have passed validation before any canonical path is replaced.
    os.replace(grid_temp,GRID)
    for temp,target in household_outputs: os.replace(temp,target)
    manifest={"backup_sha256":backup_hashes,"completed_sha256":{p.name:sha256(p) for p in [GRID,*HOUSEHOLDS]},
              "boundary_policy":"nearest valid observation (forward/back fill); invoked only at series edges",
              "grid":grid_report,"households":household_report}
    (BACKUP/"completion_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"backup":str(BACKUP),"grid":grid_report,"households":household_report},indent=2))


if __name__=="__main__": main()
