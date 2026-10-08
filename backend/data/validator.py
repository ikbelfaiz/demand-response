"""Dataset-wide scientific and relational validation."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from backend.config import get_v3_paths
from .schema import ValidationReport
from .v3_loader import load_events, load_grid, load_household_info, load_participation


def _report(name: str, frame: pd.DataFrame, key: list[str], timestamp: str | None = None) -> ValidationReport:
    parsed = pd.to_datetime(frame[timestamp], errors="coerce") if timestamp else None
    invalid = int(parsed.isna().sum()) if parsed is not None else 0
    return ValidationReport(
        source=name, row_count=len(frame), start=parsed.min() if parsed is not None else None,
        end=parsed.max() if parsed is not None else None,
        duplicate_keys=int(frame.duplicated(key).sum()),
        missing_values={c: int(v) for c, v in frame.isna().sum().items() if v}, invalid_records=invalid,
    )


def parquet_inventory(path: Path) -> dict[str, object]:
    meta = pq.ParquetFile(path).metadata
    return {"rows": meta.num_rows, "row_groups": meta.num_row_groups,
            "columns": [meta.schema.column(i).name for i in range(meta.num_columns)]}


def validate_v3() -> dict[str, object]:
    paths = get_v3_paths()
    grid, info, events, participation = load_grid(), load_household_info(), load_events(), load_participation()
    ids = set(info.client_id)
    event_ids = set(events.event_id)
    relationship_errors = {
        "participation_unknown_households": int((~participation.client_id.isin(ids)).sum()),
        "participation_unknown_events": int((~participation.event_id.isin(event_ids)).sum()),
        "missing_event_household_pairs": len(ids) * len(event_ids) - len(participation),
    }
    return {
        "reports": [
            _report("grid_1min.csv", grid, ["timestamp"], "timestamp"),
            _report("households_info.csv", info, ["client_id"]),
            _report("dr_events.csv", events, ["event_id"], "start_ts"),
            _report("dr_participation.csv", participation, ["event_id", "client_id"]),
        ],
        "parquet": {p.name: parquet_inventory(p) for p in paths.households},
        "relationships": relationship_errors,
        "household_count": len(ids), "event_count": len(event_ids),
        "submeter_count": int(info.has_submeter.sum()),
        "grid_expected_minutes": 365 * 24 * 60,
        "grid_complete_timeline": len(grid) == 365 * 24 * 60 and not grid.timestamp.duplicated().any(),
    }
