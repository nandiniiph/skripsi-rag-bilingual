from __future__ import annotations
from pathlib import Path
import pandas as pd
from .common import ensure_parent


def build_knowledge_bases(en_chunks: pd.DataFrame, id_chunks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    en = en_chunks[["parent_id","content_id","content","source","focus_area"]].copy()
    en.insert(0, "doc_id", en["content_id"] + "-en"); en.insert(3, "language", "en")
    idf = id_chunks[["parent_id","content_id","content","source","focus_area"]].copy()
    idf.insert(0, "doc_id", idf["content_id"] + "-id"); idf.insert(3, "language", "id")
    cols=["doc_id","content_id","parent_id","language","content","source","focus_area"]
    return en[cols], idf[cols]


def save_kb(df: pd.DataFrame, path: Path):
    ensure_parent(path); df.to_csv(path, index=False)
