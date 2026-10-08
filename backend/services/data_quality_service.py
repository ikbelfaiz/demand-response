"""Dataset inventory and consistency summaries for the canonical v3 source."""
import pandas as pd
import pyarrow.parquet as pq
from backend.config import PROJECT_ROOT, get_v3_paths
from backend.data.validator import validate_v3


def source_inventory() -> list[dict[str,object]]:
    rows=[]
    for name,path in get_v3_paths().inventory().items():
        rows.append({"file":name,"relative_path":str(path.relative_to(PROJECT_ROOT)),"exists":path.is_file(),
                     "size_mb":round(path.stat().st_size/1_048_576,2) if path.is_file() else None})
    return rows


def channel_catalog(grid:pd.DataFrame)->pd.DataFrame:
    rows=[{"scope":"Community grid","channel":c,"dtype":str(grid[c].dtype)} for c in grid.columns]
    schema=pq.ParquetFile(get_v3_paths().households[0]).schema_arrow
    rows.extend({"scope":"Household meters","channel":field.name,"dtype":str(field.type)} for field in schema)
    return pd.DataFrame(rows)


__all__=["source_inventory","channel_catalog","validate_v3"]
