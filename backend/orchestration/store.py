"""DR orchestrator storage (SQLAlchemy 2.0; SQLite by default, PostgreSQL in production)."""
from datetime import datetime

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, String, Text, JSON,
                        UniqueConstraint, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from backend.orchestration import settings as config


class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------ households
class Client(Base):
    __tablename__ = "clients"
    client_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    region_id: Mapped[str] = mapped_column(String(16), default=config.REGION)
    contracted_kva: Mapped[float] = mapped_column(Float)
    segment: Mapped[str] = mapped_column(String(64), default="Unknown")
    n_occupants: Mapped[int] = mapped_column(Integer, default=0)
    has_ac: Mapped[bool] = mapped_column(Boolean, default=False)
    has_electric_water_heater: Mapped[bool] = mapped_column(Boolean, default=False)
    has_washing_machine: Mapped[bool] = mapped_column(Boolean, default=False)


class UserPreference(Base):
    __tablename__ = "user_preferences"
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.client_id"), primary_key=True)
    language: Mapped[str] = mapped_column(String(2), default=config.DEFAULT_LANGUAGE)   # fr | ar | en
    channel: Mapped[str] = mapped_column(String(8), default="app")                      # app | sms
    quiet_start: Mapped[str] = mapped_column(String(5), default="22:00")
    quiet_end: Mapped[str] = mapped_column(String(5), default="07:00")
    reminders_on: Mapped[bool] = mapped_column(Boolean, default=True)


class AppliancePreference(Base):
    __tablename__ = "appliance_preferences"
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.client_id"), primary_key=True)
    device: Mapped[str] = mapped_column(String(32), primary_key=True)                  # ac | water_heater | washing_machine
    allow_recommendation: Mapped[bool] = mapped_column(Boolean, default=True)


# ------------------------------------------------------------------ tariffs (seeded from pricing.py)
class TariffBand(Base):
    __tablename__ = "tariff_bands"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    economic: Mapped[bool] = mapped_column(Boolean)
    max_kwh_month: Mapped[float] = mapped_column(Float)          # inf stored as 1e9
    price_mill_kwh: Mapped[int] = mapped_column(Integer)


class SurchargeLevel(Base):
    __tablename__ = "surcharge_levels"
    level: Mapped[int] = mapped_column(Integer, primary_key=True)
    surcharge: Mapped[float] = mapped_column(Float)
    description: Mapped[str] = mapped_column(String(128))


class TariffParam(Base):
    __tablename__ = "tariff_params"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String(32))


# ------------------------------------------------------------------ forecasts (input from the DR forecast model)
class Forecast(Base):
    __tablename__ = "forecasts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime)
    target_ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    predicted_demand_kw: Mapped[float] = mapped_column(Float)
    predicted_supply_kw: Mapped[float] = mapped_column(Float)
    predicted_margin_kw: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64))


# ------------------------------------------------------------------ events
class DREvent(Base):
    __tablename__ = "dr_events"
    event_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    region_id: Mapped[str] = mapped_column(String(16), default=config.REGION)
    start_ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    end_ts: Mapped[datetime] = mapped_column(DateTime)
    level: Mapped[int] = mapped_column(ForeignKey("surcharge_levels.level"))
    # proposed → confirmed → notified → active → done → billed   |   vetoed | cancelled
    status: Mapped[str] = mapped_column(String(16), default="proposed", index=True)
    max_deficit_kw: Mapped[float] = mapped_column(Float)
    shortage_energy_kwh: Mapped[float] = mapped_column(Float)
    deficit_share: Mapped[float] = mapped_column(Float)
    recommended_reduction_pct: Mapped[float] = mapped_column(Float)
    forecast_model_version: Mapped[str] = mapped_column(String(64))
    decision_policy_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    confirmed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    cancel_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    targets: Mapped[list["EventTarget"]] = relationship(back_populates="event", cascade="all, delete-orphan")


class EventTarget(Base):
    """One row per household and event: what was planned, sent and answered."""
    __tablename__ = "event_targets"
    __table_args__ = (UniqueConstraint("event_id", "client_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("dr_events.event_id"), index=True)
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.client_id"), index=True)
    rank: Mapped[int] = mapped_column(Integer)
    segment: Mapped[str] = mapped_column(String(64))
    usual_window_kwh: Mapped[float] = mapped_column(Float)
    projected_month_kwh: Mapped[float] = mapped_column(Float)
    band_price: Mapped[int] = mapped_column(Integer)
    peak_price: Mapped[int] = mapped_column(Integer)
    exempt: Mapped[bool] = mapped_column(Boolean)               # economic band
    personalised: Mapped[bool] = mapped_column(Boolean)         # device recommendation reliable enough
    recommended_device: Mapped[str | None] = mapped_column(String(32), nullable=True)
    action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    device_kwh: Mapped[float | None] = mapped_column(Float, nullable=True)
    device_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_saving_dt: Mapped[float] = mapped_column(Float, default=0.0)
    expected_surcharge_dt: Mapped[float] = mapped_column(Float, default=0.0)
    response: Mapped[str] = mapped_column(String(16), default="no_response")   # accept | decline | later | no_response
    response_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    event: Mapped[DREvent] = relationship(back_populates="targets")


class Notification(Base):
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.client_id"), index=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("dr_events.event_id"), nullable=True)
    kind: Mapped[str] = mapped_column(String(32))     # alert_personal | alert_general | alert_exempt | reminder | summary | cancellation
    language: Mapped[str] = mapped_column(String(2))
    channel: Mapped[str] = mapped_column(String(8))
    text: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON)
    sent_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class EventBill(Base):
    """Real surcharge of one event (equation 7), computed the day after with the band projected at that time."""
    __tablename__ = "event_bills"
    __table_args__ = (UniqueConstraint("event_id", "client_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("dr_events.event_id"), index=True)
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.client_id"), index=True)
    window_kwh: Mapped[float] = mapped_column(Float)
    missing_slots: Mapped[int] = mapped_column(Integer)
    band_price: Mapped[int] = mapped_column(Integer)
    level: Mapped[int] = mapped_column(Integer)
    window_cost_dt: Mapped[float] = mapped_column(Float)
    surcharge_dt: Mapped[float] = mapped_column(Float)
    provisional: Mapped[bool] = mapped_column(Boolean, default=True)   # final amount set by the monthly bill


class MonthlyBill(Base):
    __tablename__ = "monthly_bills"
    __table_args__ = (UniqueConstraint("client_id", "month"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.client_id"), index=True)
    month: Mapped[str] = mapped_column(String(7))      # YYYY-MM
    monthly_kwh: Mapped[float] = mapped_column(Float)
    band_price: Mapped[int] = mapped_column(Integer)
    energy_dt: Mapped[float] = mapped_column(Float)
    dr_surcharge_raw_dt: Mapped[float] = mapped_column(Float)
    dr_surcharge_dt: Mapped[float] = mapped_column(Float)
    cap_applied: Mapped[bool] = mapped_column(Boolean)
    power_fee_dt: Mapped[float] = mapped_column(Float)
    total_before_tax_dt: Mapped[float] = mapped_column(Float)


def get_engine(url: str | None = None):
    return create_engine(url or config.DATABASE_URL, future=True)


def get_sessionmaker(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


# ------------------------------------------------------------------ creation + seeding
import numpy as np  # noqa: E402

from backend.intelligence import pricing  # noqa: E402
from backend.orchestration.data_access import households  # noqa: E402

LEVEL_TEXT = {1: "Small expected deficit", 2: "Default for DR events",
              3: "Critical event: production outage during a heatwave"}


def seed(session, languages_seed: int = 7):
    for limit, price in pricing.ECONOMIC_BANDS:
        session.add(TariffBand(economic=True, max_kwh_month=limit, price_mill_kwh=price))
    for limit, price in pricing.NORMAL_BANDS:
        session.add(TariffBand(economic=False, max_kwh_month=min(limit, 1e9), price_mill_kwh=price))
    for lvl, s in pricing.LEVELS.items():
        session.add(SurchargeLevel(level=lvl, surcharge=s, description=LEVEL_TEXT[lvl]))
    session.add_all([
        TariffParam(name="power_fee", value=pricing.POWER_FEE_DT_PER_KVA * 1000, unit="mill/kVA/month"),
        TariffParam(name="municipal_surcharge", value=5, unit="mill/kWh"),
        TariffParam(name="monthly_cap_share", value=pricing.MONTHLY_CAP_SHARE, unit="share of energy cost"),
    ])
    rng = np.random.default_rng(languages_seed)
    for cid, r in households().iterrows():
        session.add(Client(client_id=cid, contracted_kva=float(r.contracted_kva), segment=r.segment,
                           n_occupants=int(r.n_occupants), has_ac=bool(r.has_ac),
                           has_electric_water_heater=bool(r.has_electric_water_heater),
                           has_washing_machine=bool(r.has_washing_machine)))
        # demo preferences: 70% French, 30% Arabic
        session.add(UserPreference(client_id=cid, language="fr" if rng.random() < 0.7 else "ar"))
        if rng.random() < 0.1:   # a few households refuse AC recommendations
            session.add(AppliancePreference(client_id=cid, device="ac", allow_recommendation=False))
    session.commit()


def init_db(url: str | None = None, reset: bool = True):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_engine(url)
    if reset:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    Session = get_sessionmaker(engine)
    with Session() as s:
        seed(s)
    return engine


