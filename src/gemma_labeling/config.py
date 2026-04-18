from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
UI_ROOT = Path(__file__).resolve().parent / "ui"

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL_ID = os.getenv("GEMMA_LABELING_MODEL_ID", "google/gemma-4-E2B-it")
DEFAULT_HOST = os.getenv("GEMMA_LABELING_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("GEMMA_LABELING_PORT", "8000"))
DEFAULT_MAX_TAGS = int(os.getenv("GEMMA_LABELING_MAX_TAGS", "12"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("GEMMA_LABELING_MAX_NEW_TOKENS", "48"))
DEFAULT_TEMPERATURE = float(os.getenv("GEMMA_LABELING_TEMPERATURE", "0.0"))
DEFAULT_REQUIRE_CUDA = os.getenv("GEMMA_LABELING_REQUIRE_CUDA", "1").lower() not in {
    "0",
    "false",
    "no",
}

_taxonomy_env = os.getenv("GEMMA_LABELING_TAXONOMY_PATH")
TAXONOMY_PATH = (
    Path(_taxonomy_env).expanduser()
    if _taxonomy_env
    else DATA_ROOT / "taxonomy" / "e621_tags.yaml"
)
