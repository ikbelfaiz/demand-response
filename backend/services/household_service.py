"""Household analytics over complete canonical half-hour readings."""
import pandas as pd
import pyarrow.dataset as ds
from backend.config import get_v3_paths


def summary(frame: pd.DataFrame) -> dict[str,float]:
    p=frame.get("aggregate_power_kw",pd.Series(dtype=float)); peak=float(p.max()); average=float(p.mean())
    return {"energy_kwh":float(frame.get("aggregate_power_energy_kwh",pd.Series(dtype=float)).sum(min_count=1)),
            "average_kw":average,"peak_kw":peak,"load_factor":average/peak if peak>0 else 0.0}


def daily_energy(frame: pd.DataFrame) -> pd.DataFrame:
    work=frame.assign(date=frame.timestamp.dt.date)
    return work.groupby("date",as_index=False).agg(energy_kwh=("aggregate_power_energy_kwh","sum"))


def monthly_energy(frame: pd.DataFrame) -> pd.DataFrame:
    work=frame.assign(month=frame.timestamp.dt.to_period("M").dt.to_timestamp())
    return work.groupby("month",as_index=False).agg(energy_kwh=("aggregate_power_energy_kwh","sum"))


def hourly_profile(frame: pd.DataFrame) -> pd.DataFrame:
    work=frame.assign(hour=frame.timestamp.dt.hour)
    return work.groupby("hour",as_index=False).agg(mean_kw=("aggregate_power_kw","mean"),p95_kw=("aggregate_power_kw",lambda x:x.quantile(.95)))


def usage_heatmap(frame: pd.DataFrame) -> pd.DataFrame:
    work=frame.assign(date=frame.timestamp.dt.date,hour=frame.timestamp.dt.hour)
    return work.pivot_table(index="date",columns="hour",values="aggregate_power_kw",aggfunc="mean")


def portfolio_summary(start:object,end:object)->pd.DataFrame:
    totals={}; expression=(ds.field("timestamp")>=pd.Timestamp(start))&(ds.field("timestamp")<pd.Timestamp(end))
    dataset=ds.dataset([str(p) for p in get_v3_paths().households],format="parquet")
    for batch in dataset.to_batches(columns=["client_id","aggregate_power_w"],filter=expression,batch_size=500_000):
        part=batch.to_pandas(); grouped=part.groupby("client_id",observed=True).aggregate(watts_sum=("aggregate_power_w","sum"),valid=("aggregate_power_w","count"),peak_w=("aggregate_power_w","max"),samples=("aggregate_power_w","size"))
        for client,row in grouped.iterrows():
            acc=totals.setdefault(str(client),{"watts_sum":0.,"valid":0,"peak_w":0.,"samples":0}); acc["watts_sum"]+=float(row.watts_sum); acc["valid"]+=int(row.valid); acc["peak_w"]=max(acc["peak_w"],float(row.peak_w)); acc["samples"]+=int(row.samples)
    rows=[{"client_id":c,"energy_kwh":a["watts_sum"]/60_000,"average_kw":a["watts_sum"]/a["valid"]/1000,"peak_kw":a["peak_w"]/1000} for c,a in totals.items()]
    return pd.DataFrame(rows).sort_values("energy_kwh",ascending=False,ignore_index=True)
