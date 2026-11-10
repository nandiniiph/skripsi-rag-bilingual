from __future__ import annotations
from pathlib import Path
import pandas as pd
from .common import ensure_parent


def sample_evaluation_pairs(qa_en: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n > len(qa_en): raise ValueError(f"Meminta {n} sampel, data hanya {len(qa_en)}")
    sample = qa_en.sample(n=n, random_state=seed).reset_index(drop=True).copy()
    sample.insert(0, "pair_id", [f"QA{i:06d}" for i in range(1, len(sample)+1)])
    return sample


def save_sample(df: pd.DataFrame, path: Path):
    ensure_parent(path); df.to_csv(path, index=False)
