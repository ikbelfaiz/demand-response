"""Source-tree bridge so the standalone package is importable by the application.

The research CLI still installs from ``nilm_research/src``.  The application runs
from the repository root, so extending this package path avoids copying any model
or preprocessing implementation into the backend.
"""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parent / "src" / "nilm_research"))
