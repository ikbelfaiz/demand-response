from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nilm_research.config import load_config, resolved_config
from nilm_research.data import NILMRepository, fit_transforms, prepare_partition


@pytest.fixture(scope="session")
def cfg():
    return resolved_config(load_config(ROOT / "configs" / "executed_cpu.json"))


@pytest.fixture(scope="session")
def prepared(cfg):
    repo = NILMRepository(cfg)
    panel = repo.panel_households()
    heldout = set(cfg["evaluation"]["held_out_households"])
    fit_homes = [x for x in panel if x not in heldout]
    transforms, provenance = fit_transforms(repo, cfg, fit_homes)
    part = prepare_partition(repo, cfg, transforms, fit_homes[0], "train")
    return repo, panel, fit_homes, transforms, provenance, part

