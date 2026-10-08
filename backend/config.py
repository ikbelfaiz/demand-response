"""Application configuration and project-relative v3 dataset paths."""
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
V3_DIR = DATA_DIR / "dr_dataset_v3_grid_et_code"


@dataclass(frozen=True)
class V3Paths:
    grid: Path = V3_DIR / "grid_1min.csv"
    households: tuple[Path, Path] = (
        DATA_DIR / "households_1min_C001_C025.parquet",
        DATA_DIR / "households_1min_C026_C050.parquet",
    )
    household_info: Path = V3_DIR / "households_info.csv"
    events: Path = V3_DIR / "dr_events.csv"
    participation: Path = V3_DIR / "dr_participation.csv"
    ground_truth: Path = V3_DIR / "ground_truth_savings.csv"
    generator: Path = V3_DIR / "generate_dataset_v3.py"
    core: Path = V3_DIR / "dr_core.py"

    def inventory(self) -> dict[str, Path]:
        return {
            "grid_1min.csv": self.grid,
            "households_1min_C001_C025.parquet": self.households[0],
            "households_1min_C026_C050.parquet": self.households[1],
            "households_info.csv": self.household_info,
            "dr_events.csv": self.events,
            "dr_participation.csv": self.participation,
            "ground_truth_savings.csv": self.ground_truth,
            "generate_dataset_v3.py": self.generator,
            "dr_core.py": self.core,
        }


V3_PATHS = V3Paths()
OPERATIONAL_FREQUENCY = "30min"
SOURCE_FREQUENCY = "1min"
EXPECTED_MINUTES_PER_INTERVAL = 30
NETWORK_LOSS_ASSUMPTION = 0.03
NON_RESIDENTIAL_SHARE = 0.35


def get_v3_paths() -> V3Paths:
    return V3_PATHS


def canonical_data_signature() -> tuple[tuple[str,int,int],...]:
    """Change whenever a canonical sensor file is replaced."""
    files=(V3_PATHS.grid,*V3_PATHS.households)
    return tuple((str(path),path.stat().st_size,path.stat().st_mtime_ns) for path in files)
