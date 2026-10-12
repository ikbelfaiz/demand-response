"""Customer profiling (final architecture).

Receives the peak window and the deficit from the D+1 forecast, decides who to
target, gives the chatbot each household's context and computes the economics
of each event. Smart meter (30 min) + temperature only.

Permanent outputs (interface)
  1. household_segments       segment per household (K-means), DR acceptance rate,
                              comparison with similar households
  2. segment_load_profiles    typical curves per segment x season x day type x slot
Before a DR event (orchestrator)
  3. event_targeting          households sorted by segment priority, then by usual
                              consumption in the peak window          -> event manager
  4. chatbot_context          segment, usual window kWh, acceptance rate, STEG band,
                              evening comparison with similar homes   -> chatbot
After a DR event (economics / pricing)
  5a. event_costs             cost of the event for each informed household (CPP +50 %)
  5b. segment_savings         savings of each segment, measured with the control group
  5c. rewards                 rewards for accepting households from their segment's savings

Segments are trained on January to mid-June (days without events); January-May alone
has only 11 hot days and gave unstable segments. This module never
reads ``ground_truth_savings.csv``: evaluation lives in ``backend.intelligence.evaluation``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SPIKE_THRESHOLD_KW = 15.0
MAX_GAP_MINUTES = 120
SLOT = "30min"
SLOT_HOURS = 0.5
TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2025-06-16")          # exclusive: January to mid-June (Jan-May unstable, see report)
EVENING_HOURS = (18, 22)
COOLING_BASE_C = 24.0
MIN_HOT_DAYS = 10
WEEKEND_DAYS = (5, 6)                           # Saturday, Sunday
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
USUAL_LOOKBACK_DAYS = 28                        # history used for "usual" consumption
PRE_EVENT_HOURS = 4
EXPECTED_REDUCTION_IF_ACCEPTED = 0.20           # reduction target of the DR events
N_CLUSTERS = 4
RANDOM_STATE = 42

# STEG residential tariff: the whole month is billed at the band reached (millimes/kWh)
STEG_BANDS = [(50, 62), (100, 96), (200, 176), (300, 218), (500, 341), (np.inf, 414)]
ECONOMIC_BAND_MAX_KWH = 100                     # never surcharged
SURCHARGE_LEVELS = {1: 0.30, 2: 0.50, 3: 1.00}  # critical peak pricing, level 2 by default
DEFAULT_SURCHARGE_LEVEL = 2

FEATURES = ["annual_kwh", "peak_kw", "base_load_kw", "load_factor", "evening_share", "cooling_kwh_per_deg"]

SEG_LARGE = "Large evening-peak households"
SEG_DAYTIME = "Daytime-occupied, high base load"
SEG_MEDIUM = "Medium AC users"
SEG_LOW = "Low consumers without AC"
PRIORITY = {SEG_LARGE: 1, SEG_DAYTIME: 2, SEG_MEDIUM: 3, SEG_LOW: 4}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def clean_household_power(raw: pd.DataFrame) -> tuple[pd.Series, int]:
    """Cleaned one-minute aggregate power (kW) and the number of spikes removed."""
    power = pd.to_numeric(raw["aggregate_power_w"], errors="coerce") / 1000.0
    power.index = pd.DatetimeIndex(raw["timestamp"])
    power = power.sort_index()
    spikes = int((power > SPIKE_THRESHOLD_KW).sum())
    power = power.mask(power > SPIKE_THRESHOLD_KW)
    step = power.index.to_series().diff().median()
    step_minutes = max(step / pd.Timedelta("1min"), 1.0) if pd.notna(step) else 1.0
    power = power.interpolate(method="time", limit=int(MAX_GAP_MINUTES / step_minutes), limit_area="inside")
    return power, spikes


def to_half_hour_kw(power_kw: pd.Series) -> pd.Series:
    """Average power per 30-minute slot (power is averaged, never summed)."""
    return power_kw.resample(SLOT).mean()


def event_days(grid30: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(grid30.index[grid30["is_dr_event"].fillna(0) > 0].normalize().unique())


def natural_days(kw30: pd.DataFrame, grid30: pd.DataFrame) -> pd.DataFrame:
    """Drop DR event days: DR changed consumption there."""
    return kw30.loc[~kw30.index.normalize().isin(event_days(grid30))]


def _window(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    return frame.loc[(frame.index >= pd.Timestamp(start)) & (frame.index < pd.Timestamp(end))]


# ---------------------------------------------------------------------------
# Features (one row per household)
# ---------------------------------------------------------------------------
def compute_features(kw30: pd.DataFrame, grid30: pd.DataFrame,
                     start=TRAIN_START, end=TRAIN_END) -> pd.DataFrame:
    """Profile features on days without events between ``start`` (incl.) and ``end`` (excl.)."""
    kw = _window(natural_days(kw30, grid30), start, end)
    kwh = kw * SLOT_HOURS
    n_days = kw.index.normalize().nunique()
    peak = kw.quantile(0.99)
    hour = kw.index.hour
    evening = kwh[(hour >= EVENING_HOURS[0]) & (hour < EVENING_HOURS[1])]

    daily_kwh = kwh.resample("D").sum(min_count=40).dropna(how="all")
    daily_t = grid30["temperature_c"].resample("D").mean().reindex(daily_kwh.index)
    hot = (daily_t > COOLING_BASE_C) & daily_kwh.notna().all(axis=1)
    slopes = {c: (float(np.polyfit(daily_t[hot], daily_kwh.loc[hot, c], 1)[0]) if hot.sum() >= MIN_HOT_DAYS else np.nan)
              for c in kw.columns}

    feats = pd.DataFrame({
        "annual_kwh": kwh.sum() / n_days * 365.0,
        "peak_kw": peak,
        "base_load_kw": kw.quantile(0.05),
        "load_factor": kw.mean() / peak,
        "evening_share": evening.sum() / kwh.sum(),
        "cooling_kwh_per_deg": pd.Series(slopes),
    })
    feats.index.name = "client_id"
    feats.attrs["hot_days"] = int(hot.sum())
    return feats


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
@dataclass
class ProfilingModel:
    """Serializable K-means model (scaler, centroids, segment names)."""
    features: list[str]
    mean: np.ndarray
    scale: np.ndarray
    centroids: np.ndarray
    names: dict[int, str]
    medians: np.ndarray
    silhouette: float

    def transform(self, feats: pd.DataFrame) -> np.ndarray:
        X = feats[self.features].to_numpy(dtype=float)
        return (np.where(np.isnan(X), self.medians, X) - self.mean) / self.scale

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        d = ((self.transform(feats)[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)

    def to_dict(self) -> dict:
        return {"features": self.features, "scaler_mean": self.mean.tolist(), "scaler_scale": self.scale.tolist(),
                "centroids": self.centroids.tolist(), "cluster_names": {str(k): v for k, v in self.names.items()},
                "feature_medians": self.medians.tolist(), "silhouette": self.silhouette}

    @classmethod
    def from_dict(cls, d: dict) -> "ProfilingModel":
        return cls(list(d["features"]), np.array(d["scaler_mean"]), np.array(d["scaler_scale"]),
                   np.array(d["centroids"]), {int(k): v for k, v in d["cluster_names"].items()},
                   np.array(d["feature_medians"]), float(d["silhouette"]))


def standardise(feats: pd.DataFrame):
    X = feats[FEATURES].to_numpy(dtype=float)
    medians = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), medians, X)
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale == 0] = 1.0
    return (X - mean) / scale, mean, scale, medians


def kmeans_labels(feats: pd.DataFrame, k: int = N_CLUSTERS) -> np.ndarray:
    from sklearn.cluster import KMeans
    X, *_ = standardise(feats)
    return KMeans(n_clusters=k, n_init=100, random_state=RANDOM_STATE).fit(X).labels_


def name_clusters(feats: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Name clusters by their discriminating feature (unique assignment):
    lowest heat sensitivity -> Low, no AC; highest base load -> Daytime-occupied;
    highest evening energy -> Large evening-peak; the rest -> Medium AC users."""
    f = feats.assign(evening_kwh=feats["annual_kwh"] * feats["evening_share"], cluster=labels)
    means = f.groupby("cluster")[FEATURES + ["evening_kwh"]].mean()
    names, remaining = {}, list(means.index)
    for column, largest, name in [("cooling_kwh_per_deg", False, SEG_LOW), ("base_load_kw", True, SEG_DAYTIME),
                                  ("evening_kwh", True, SEG_LARGE), ("evening_kwh", True, SEG_MEDIUM)]:
        if not remaining:
            break
        sub = means.loc[remaining, column]
        chosen = int(sub.idxmax() if largest else sub.idxmin())
        names[chosen] = name
        remaining.remove(chosen)
    for c in remaining:
        names[int(c)] = f"Segment {c}"
    return names


def fit(feats: pd.DataFrame, k: int = N_CLUSTERS) -> tuple[ProfilingModel, np.ndarray]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X, mean, scale, medians = standardise(feats)
    km = KMeans(n_clusters=k, n_init=100, random_state=RANDOM_STATE).fit(X)
    model = ProfilingModel(list(FEATURES), mean, scale, km.cluster_centers_, name_clusters(feats, km.labels_),
                           medians, float(silhouette_score(X, km.labels_)))
    return model, km.labels_


# ---------------------------------------------------------------------------
# Helpers shared by several outputs
# ---------------------------------------------------------------------------
def acceptance_rates(participation: pd.DataFrame, events: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Share of past events each household accepted, using events that ended before ``as_of``."""
    ev = events if as_of is None else events.loc[pd.to_datetime(events["end_ts"]) <= pd.Timestamp(as_of)]
    p = participation.loc[participation["event_id"].isin(ev["event_id"])]
    out = p.assign(accepted=p["response"].eq("accept")).groupby("client_id").agg(
        acceptance_rate=("accepted", "mean"), n_invitations=("accepted", "size"))
    return out


def steg_band(monthly_kwh: float) -> int:
    """Price (millimes/kWh) of the band reached; the whole month is billed at that price."""
    for limit, price in STEG_BANDS:
        if monthly_kwh <= limit:
            return price
    return STEG_BANDS[-1][1]


def month_position(kw30: pd.DataFrame, as_of) -> pd.DataFrame:
    """Month-to-date kWh before ``as_of`` and the projected month-end kWh, with their bands."""
    as_of = pd.Timestamp(as_of)
    month_start = as_of.normalize().replace(day=1)
    days_in_month = month_start.days_in_month
    elapsed = (as_of - month_start) / pd.Timedelta("1D")
    mtd = _window(kw30, month_start, as_of).sum() * SLOT_HOURS
    if elapsed >= 1:
        projected = mtd / elapsed * days_in_month
    else:   # first day of the month: use the previous 7 days' daily average
        projected = _window(kw30, as_of - pd.Timedelta(days=7), as_of).sum() * SLOT_HOURS / 7 * days_in_month
    out = pd.DataFrame({"month_to_date_kwh": mtd, "projected_month_kwh": projected})
    out["steg_band_millimes"] = out["month_to_date_kwh"].map(steg_band)
    out["projected_band_millimes"] = out["projected_month_kwh"].map(steg_band)
    return out


def usual_window_kwh(kw30: pd.DataFrame, grid30: pd.DataFrame, as_of, peak_start, peak_end) -> pd.Series:
    """Average kWh each household uses in the peak window's slots on recent days without events."""
    hist = _window(natural_days(kw30, grid30), pd.Timestamp(as_of) - pd.Timedelta(days=USUAL_LOOKBACK_DAYS), as_of)
    start, end = pd.Timestamp(peak_start), pd.Timestamp(peak_end)
    tod = hist.index - hist.index.normalize()
    s, e = start - start.normalize(), end - start.normalize()
    if e <= pd.Timedelta(days=1):
        in_window = (tod >= s) & (tod < e)
    else:   # window crossing midnight
        in_window = (tod >= s) | (tod < e - pd.Timedelta(days=1))
    days = hist.index.normalize().nunique()
    return hist.loc[in_window].sum() * SLOT_HOURS / max(days, 1)


def usual_evening_kwh(kw30: pd.DataFrame, grid30: pd.DataFrame, as_of) -> pd.Series:
    hist = _window(natural_days(kw30, grid30), pd.Timestamp(as_of) - pd.Timedelta(days=USUAL_LOOKBACK_DAYS), as_of)
    h = hist.index.hour
    days = hist.index.normalize().nunique()
    return hist.loc[(h >= EVENING_HOURS[0]) & (h < EVENING_HOURS[1])].sum() * SLOT_HOURS / max(days, 1)


def comparison_message(pct: float) -> str:
    if pd.isna(pct):
        return ""
    if abs(pct) < 5:
        return "In the evening you use about the same as households like yours."
    return f"You use {abs(pct):.0f}% {'more' if pct > 0 else 'less'} than households like yours in the evening."


def _evening_comparison(segment: pd.Series, evening: pd.Series) -> pd.DataFrame:
    e = evening.reindex(segment.index)
    median = e.groupby(segment).transform("median")
    pct = 100 * (e / median - 1)
    return pd.DataFrame({"usual_evening_kwh": e, "segment_median_evening_kwh": median,
                         "vs_similar_households_pct": pct, "comparison_message": pct.map(comparison_message)})


# ---------------------------------------------------------------------------
# Permanent outputs (interface)
# ---------------------------------------------------------------------------
def household_segments(feats: pd.DataFrame, labels: np.ndarray, model: ProfilingModel,
                       acceptance: pd.DataFrame, kw30: pd.DataFrame, grid30: pd.DataFrame, as_of) -> pd.DataFrame:
    """Output 1: segment, priority, acceptance rate and evening comparison per household."""
    out = feats.copy()
    out["cluster"] = labels
    out["segment"] = out["cluster"].map(model.names)
    out["dr_priority"] = out["segment"].map(PRIORITY).fillna(5).astype(int)
    out = out.join(acceptance[["acceptance_rate", "n_invitations"]])
    out = out.join(_evening_comparison(out["segment"], usual_evening_kwh(kw30, grid30, as_of)))
    out["as_of"] = pd.Timestamp(as_of)
    return out.reset_index().sort_values(["dr_priority", "client_id"], ignore_index=True)


def segment_load_profiles(segments: pd.DataFrame, kw30: pd.DataFrame, grid30: pd.DataFrame) -> pd.DataFrame:
    """Output 2: average kW per household, by segment x season x day type x half-hour slot."""
    nat = natural_days(kw30, grid30)
    idx = nat.index
    keys = [pd.Index(idx.month).map(SEASONS).to_numpy(),
            np.where(np.isin(idx.dayofweek, WEEKEND_DAYS), "weekend", "weekday"),
            idx.hour * 2 + idx.minute // 30]
    seg_of = segments.set_index("client_id")["segment"]
    rows = []
    for seg, members in seg_of.groupby(seg_of):
        per_home = nat[members.index.intersection(nat.columns)].mean(axis=1)
        for (season, day_type, slot), v in per_home.groupby(keys).mean().items():
            rows.append({"segment": seg, "season": season, "day_type": day_type, "slot": int(slot),
                         "mean_kw_per_household": float(v)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Before a DR event (orchestrator)
# ---------------------------------------------------------------------------
def event_targeting(segments: pd.DataFrame, kw30: pd.DataFrame, grid30: pd.DataFrame,
                    acceptance: pd.DataFrame, peak_start, peak_end, deficit_kwh: float, as_of) -> pd.DataFrame:
    """Output 3: households sorted by segment priority, then by usual consumption in the peak window.

    The cumulative expected reduction tells the event manager how far down the list
    it must go to cover the forecast deficit; it decides who is notified.
    """
    window = usual_window_kwh(kw30, grid30, as_of, peak_start, peak_end)
    t = segments[["client_id", "segment", "dr_priority"]].copy()
    t["usual_window_kwh"] = t["client_id"].map(window)
    t["acceptance_rate"] = t["client_id"].map(acceptance["acceptance_rate"]).fillna(acceptance["acceptance_rate"].mean())
    t["expected_reduction_kwh"] = t["usual_window_kwh"] * t["acceptance_rate"] * EXPECTED_REDUCTION_IF_ACCEPTED
    t = t.sort_values(["dr_priority", "usual_window_kwh"], ascending=[True, False], ignore_index=True)
    t.insert(0, "rank", np.arange(1, len(t) + 1))
    t["cumulative_expected_reduction_kwh"] = t["expected_reduction_kwh"].cumsum()
    t["deficit_kwh"] = deficit_kwh
    t["covers_deficit_at_rank"] = t["cumulative_expected_reduction_kwh"] >= deficit_kwh
    t["peak_start"], t["peak_end"] = pd.Timestamp(peak_start), pd.Timestamp(peak_end)
    return t


def chatbot_context(segments: pd.DataFrame, kw30: pd.DataFrame, grid30: pd.DataFrame,
                    acceptance: pd.DataFrame, peak_start, peak_end, as_of) -> pd.DataFrame:
    """Output 4: segment, usual window kWh, acceptance rate, STEG band, evening comparison."""
    seg = segments.set_index("client_id")["segment"]
    ctx = pd.DataFrame({"segment": seg})
    ctx["usual_window_kwh"] = usual_window_kwh(kw30, grid30, as_of, peak_start, peak_end)
    ctx["acceptance_rate"] = acceptance["acceptance_rate"].reindex(ctx.index)
    ctx = ctx.join(month_position(kw30, as_of)).join(_evening_comparison(seg, usual_evening_kwh(kw30, grid30, as_of)))
    ctx["peak_start"], ctx["peak_end"] = pd.Timestamp(peak_start), pd.Timestamp(peak_end)
    ctx.index.name = "client_id"
    return ctx.reset_index()


# ---------------------------------------------------------------------------
# After a DR event: economics (pricing)
# ---------------------------------------------------------------------------
def event_costs(kw30: pd.DataFrame, event, participation: pd.DataFrame, control_group=(),
                surcharge_level: int = DEFAULT_SURCHARGE_LEVEL) -> pd.DataFrame:
    """Output 5a: cost of the event for each household = window kWh x peak price.

    Peak price = band price x (1 + surcharge). Control-group households (not
    informed) and households in the economic band (<= 100 kWh/month) are never surcharged.
    """
    start, end = pd.Timestamp(event.start_ts), pd.Timestamp(event.end_ts)
    window = _window(kw30, start, end).sum() * SLOT_HOURS
    month = month_position(kw30, end)
    surcharge = SURCHARGE_LEVELS[surcharge_level]
    resp = participation.loc[participation["event_id"].eq(event.event_id)].set_index("client_id")["response"]
    out = pd.DataFrame({"window_kwh": window}).join(month[["projected_month_kwh", "projected_band_millimes"]])
    out["response"] = resp.reindex(out.index)
    out["informed"] = ~out.index.isin(list(control_group))
    out["economic_band"] = out["projected_month_kwh"] <= ECONOMIC_BAND_MAX_KWH
    applies = out["informed"] & ~out["economic_band"]
    out["peak_price_millimes"] = out["projected_band_millimes"] * np.where(applies, 1 + surcharge, 1.0)
    out["event_cost_dt"] = out["window_kwh"] * out["peak_price_millimes"] / 1000
    out["surcharge_dt"] = out["window_kwh"] * out["projected_band_millimes"] * np.where(applies, surcharge, 0.0) / 1000
    out.insert(0, "event_id", event.event_id)
    out.index.name = "client_id"
    return out.reset_index()


SAVINGS_COLUMNS = ["event_id", "start_ts", "segment", "n_accepted", "n_comparison", "comparison_source",
                   "expected_kwh", "actual_kwh", "savings_kwh", "reduction_pct"]


def segment_savings(kw30: pd.DataFrame, segments: pd.DataFrame, events: pd.DataFrame,
                    participation: pd.DataFrame, control_groups: dict | None = None,
                    use_segments: bool = True) -> pd.DataFrame:
    """Output 5b: savings of each segment, measured with the control group.

    Comparison group = the event's control group in that segment when the event
    manager set one aside; otherwise the non-accepting households of the segment.
        expected = (comparison event kWh / comparison kWh in the 4 h before) x accepters' pre-event kWh
        savings  = expected - accepters' actual kWh
    """
    kwh = kw30 * SLOT_HOURS
    seg_of = (segments.set_index("client_id")["segment"] if use_segments
              else pd.Series("All households", index=kwh.columns))
    control_groups = control_groups or {}
    rows = []
    for ev in events.itertuples():
        start, end = pd.Timestamp(ev.start_ts), pd.Timestamp(ev.end_ts)
        win = _window(kwh, start, end).sum(min_count=1)
        pre = _window(kwh, start - pd.Timedelta(hours=PRE_EVENT_HOURS), start).sum(min_count=1)
        resp = participation.loc[participation["event_id"] == ev.event_id].set_index("client_id")["response"]
        accepted = resp.index[resp.eq("accept")].intersection(kwh.columns)
        control = pd.Index(control_groups.get(ev.event_id, [])).intersection(kwh.columns)
        others = control if len(control) else resp.index[~resp.eq("accept")].intersection(kwh.columns)
        source = "control_group" if len(control) else "non_accepting"
        if accepted.empty:
            continue
        event_ratio = win[others].sum() / pre[others].sum() if len(others) and pre[others].sum() > 0 else np.nan
        for seg in pd.unique(seg_of.reindex(accepted)):
            members = accepted[seg_of.reindex(accepted).eq(seg).to_numpy()]
            peers = others[seg_of.reindex(others).eq(seg).to_numpy()]
            ratio = win[peers].sum() / pre[peers].sum() if len(peers) and pre[peers].sum() > 0 else event_ratio
            expected, actual = ratio * pre[members].sum(), win[members].sum()
            rows.append({"event_id": ev.event_id, "start_ts": start, "segment": seg, "n_accepted": len(members),
                         "n_comparison": len(peers), "comparison_source": source if len(peers) else "event_fallback",
                         "expected_kwh": expected, "actual_kwh": actual, "savings_kwh": expected - actual,
                         "reduction_pct": 100 * (expected - actual) / expected if expected else np.nan})
    return pd.DataFrame(rows, columns=SAVINGS_COLUMNS)


def rewards(savings: pd.DataFrame, segments: pd.DataFrame, participation: pd.DataFrame,
            costs: pd.DataFrame, surcharge_level: int = DEFAULT_SURCHARGE_LEVEL) -> pd.DataFrame:
    """Output 5c: reward for each accepting household, based on its segment's savings.

    Each accepter receives an equal share of its segment's saved kWh (negative
    savings give no reward), paid at the surcharge it would have avoided:
        reward = share kWh x band price x surcharge
    """
    seg_of = segments.set_index("client_id")["segment"]
    acc = participation.loc[participation["response"].eq("accept"), ["event_id", "client_id"]].copy()
    acc["segment"] = acc["client_id"].map(seg_of)
    s = savings.assign(share_kwh=(savings["savings_kwh"].clip(lower=0) / savings["n_accepted"]))
    out = acc.merge(s[["event_id", "segment", "share_kwh", "reduction_pct"]], on=["event_id", "segment"], how="inner")
    out = out.merge(costs[["event_id", "client_id", "projected_band_millimes", "economic_band"]],
                    on=["event_id", "client_id"], how="left")
    out["reward_dt"] = out["share_kwh"] * out["projected_band_millimes"] * SURCHARGE_LEVELS[surcharge_level] / 1000
    return out.rename(columns={"share_kwh": "credited_kwh", "reduction_pct": "segment_reduction_pct"})
