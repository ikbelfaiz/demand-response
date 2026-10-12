"""Turn the priority list (customer profiling) + device recommendations (NILM) into one EventTarget per household.

Every household of the zone is informed (no control group). The priority list sets the sending
order (rank) and the level of personalisation; the pricing engine sets prices and estimates.
"""
import pandas as pd

from backend.intelligence import pricing
from backend.orchestration import settings as config
from backend.orchestration.adapters.devices import default_recommender
from backend.orchestration.store import DREvent, EventTarget


def _opt(x):
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else x


def build_targets(event: DREvent, priority: pd.DataFrame, households: pd.DataFrame,
                  refused: dict[str, set], mean_temp_c: float, recommender=None,
                  asof: pd.Timestamp | None = None) -> list[EventTarget]:
    recommender = recommender or default_recommender()
    window_hours = (event.end_ts - event.start_ts).total_seconds() / 3600
    out = []
    for r in priority.itertuples(index=False):
        band = pricing.band_price(r.projected_month_kwh, r.contracted_kva)
        rec = recommender.recommend(households.loc[r.client_id], r.usual_window_kwh, window_hours,
                                    mean_temp_c, refused.get(r.client_id, set()), client_id=r.client_id,
                                    start=pd.Timestamp(event.start_ts), end=pd.Timestamp(event.end_ts), asof=asof)
        personalised = rec.device is not None and rec.confidence >= config.DEVICE_RULE_MIN_CONFIDENCE
        out.append(EventTarget(
            event_id=event.event_id, client_id=r.client_id, rank=int(r.rank), segment=r.segment,
            usual_window_kwh=float(r.usual_window_kwh), projected_month_kwh=float(r.projected_month_kwh),
            band_price=band.price, peak_price=pricing.peak_price(band, event.level), exempt=band.economic,
            personalised=personalised,
            recommended_device=rec.device if personalised else None,
            action=rec.action if personalised else None,
            device_kwh=rec.device_kwh if personalised else None,
            device_confidence=rec.confidence,
            expected_saving_dt=pricing.expected_saving(rec.device_kwh, rec.action, band, event.level)
            if personalised else 0.0,
            expected_surcharge_dt=pricing.event_surcharge(r.usual_window_kwh, band, event.level),
            acceptance_rate=_opt(getattr(r, "acceptance_rate", None)),
            expected_reduction_kwh=_opt(getattr(r, "expected_reduction_kwh", None)),
            covers_deficit=bool(getattr(r, "covers_deficit_at_rank", False)),
            comparison_message=_opt(getattr(r, "comparison_message", None)),
        ))
    return out
