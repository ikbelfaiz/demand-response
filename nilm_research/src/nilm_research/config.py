from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    cfg = json.loads(source.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(source)
    cfg["_research_root"] = str(ROOT)
    cfg["_repo_root"] = str(REPO_ROOT)
    return cfg


def resolved_config(cfg: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(cfg)
    repo = Path(out["_repo_root"])
    research = Path(out["_research_root"])
    for key, value in list(out["data"].items()):
        if key.endswith("_path") or key.endswith("_paths"):
            if isinstance(value, list):
                out["data"][key] = [str((repo / x).resolve()) for x in value]
            else:
                out["data"][key] = str((repo / value).resolve())
    out["run"]["output_dir"] = str((research / out["run"]["output_dir"]).resolve())
    return out


def save_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")

