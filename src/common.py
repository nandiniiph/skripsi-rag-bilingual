from __future__ import annotations
import json, logging, random
from pathlib import Path
from typing import Any
import numpy as np
import yaml

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config/config.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(log_path: Path | None = None) -> logging.Logger:
    logger = logging.getLogger("nandini")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        ensure_parent(log_path)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def save_json(data: Any, path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
