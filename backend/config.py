import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "samples"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REGISTRY_PATH = BACKEND_DIR / "registry.yaml"

# Ensure runtime directories exist
try:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

class Settings(BaseModel):
    app_name: str = "SatQuery AI"
    version: str = "1.0.0"
    debug: bool = True
    registry_file: Path = REGISTRY_PATH
    sample_dir: Path = SAMPLE_DIR
    reports_dir: Path = REPORTS_DIR
    max_upload_size_mb: int = 50
    default_device: str = "cpu"

settings = Settings()
