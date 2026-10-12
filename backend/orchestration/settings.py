"""DR orchestrator settings. Data paths come from backend.config (one source of truth)."""
import os

from backend.config import DATA_DIR, PROJECT_ROOT

OUT_DIR = PROJECT_ROOT / "outputs" / "orchestrator"
DATABASE_URL = os.getenv("DR_DATABASE_URL", f"sqlite:///{OUT_DIR / 'orchestrator.db'}")
METER_CACHE = OUT_DIR / "meter_30min.parquet"
PROFILES_FILE = DATA_DIR / "customer_profiles.csv"      # output of the customer profiling module

REGION = "TUN"
SLOT = "30min"
SLOT_HOURS = 0.5
N_HOUSEHOLDS = 50

# ---- event decision (the window comes from backend.intelligence.decision_engine) ----
MAX_EVENTS_PER_WEEK = 3
LEVEL3_DEFICIT_SHARE = 0.15   # level 3 (+100%) from 15% of supply, or on a reported outage
LEVEL1_DEFICIT_SHARE = 0.05   # level 1 (+30%) below 5%; level 2 (+50%) in between

# ---- daily timetable (local time) ----
DECISION_TIME = "14:30"       # D-1: community forecast issued at 14:00 → event proposed
ALERT_TIME = "18:00"          # D-1: alerts sent
LATER_TIME = "21:00"          # D-1: second message for "Later"
BILLING_TIME = "09:00"        # D+1: bill + summary of the previous day's event
MONTH_CLOSE_TIME = "06:00"
MONTH_CLOSE_DAY = 3           # monthly bills of the previous month, on the 3rd

# ---- personalisation ----
DEVICE_RULE_MIN_CONFIDENCE = 0.4
USUAL_WINDOW_LOOKBACK_DAYS = 14
DEFAULT_LANGUAGE = "fr"
