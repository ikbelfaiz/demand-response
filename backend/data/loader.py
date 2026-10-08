from pathlib import Path
import pandas as pd

from backend.config import DatasetConfig


def load_dataset(config: DatasetConfig, *, nrows: int | None = None) -> pd.DataFrame:
    """Load an immutable raw copy; semantic conversion happens in preprocessing."""
    path = Path(config.path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, nrows=nrows, low_memory=False)

