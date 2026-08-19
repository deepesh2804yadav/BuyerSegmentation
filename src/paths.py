"""Project path helpers."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
FIGURES = ARTIFACTS / "figures"


def ensure_output_dirs() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
