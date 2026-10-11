"""Replay the DR orchestrator over a period: python scripts/run_orchestrator_replay.py --start 2025-06-01 --end 2025-09-30"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.orchestration.replay import main  # noqa: E402

if __name__ == "__main__":
    main()
