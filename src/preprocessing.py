from __future__ import annotations
import html
import re
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from .common import ensure_parent, save_json

# Pola yang benar-benar bersifat navigasi/video pada MedQuAD. Pola sengaja konservatif
# agar informasi medis tidak terhapus hanya karena mengandung kata "learn more".
NAVIGATION_PATTERNS = [
    re.compile(r"\(\s*Watch\b.*?keyboard\.\)\s*", re.I | re.S),
    re.compile(r"\(\s*Watch\s+(?:the|this|an?)\s+(?:animated\s+)?video[^)]*\)\s*", re.I | re.S),
    re.compile(r"\bSee this graphic for a quick overview.*?(?:how to learn more\.|learn more\.)", re.I | re.S),
    re.compile(r"\bSee a glossary of [^.]*?terms\.\s*", re.I),
    re.compile(r"\bClick here\s+(?:for|to)\s+[^.]*\.\s*", re.I),
]
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
SPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = html.unescape(str(value))
    # HTML -> plain text
    if re.search(r"<[^>]+>", text):
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    # Hapus URL; label manusia tetap dipertahankan
    text = URL_RE.sub(" ", text)
    # Hapus pola navigasi yang spesifik
    for pat in NAVIGATION_PATTERNS:
        text = pat.sub(" ", text)
    # Bersihkan karakter kontrol, tetapi pertahankan tanda baca medis umum
    text = "".join(ch if (ch.isprintable() or ch in "\n\t") else " " for ch in text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def normalize_for_duplicate(text: object) -> str:
    text = clean_text(text)
    text = text.casefold()
    text = SPACE_RE.sub(" ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    start = len(df)
    work = df.copy()
    for col in ["question", "answer", "source", "focus_area"]:
        if col not in work.columns:
            raise ValueError(f"Kolom wajib tidak ditemukan: {col}")

    # Question dan answer saja yang wajib lengkap sesuai metodologi.
    raw_q = work["question"].fillna("").astype(str).str.strip()
    raw_a = work["answer"].fillna("").astype(str).str.strip()
    incomplete = raw_q.eq("") | raw_a.eq("")
    removed_incomplete = int(incomplete.sum())
    work = work.loc[~incomplete].copy()

    work["question"] = work["question"].map(clean_text)
    work["answer"] = work["answer"].map(clean_text)
    work["source"] = work["source"].fillna("").astype(str).str.strip()
    work["focus_area"] = work["focus_area"].fillna("").map(clean_text)

    # Setelah cleaning, cek kembali apakah menjadi kosong.
    empty_after = work["question"].str.strip().eq("") | work["answer"].str.strip().eq("")
    removed_empty_after_clean = int(empty_after.sum())
    work = work.loc[~empty_after].copy()

    work["_norm_question"] = work["question"].map(normalize_for_duplicate)
    work["_norm_answer"] = work["answer"].map(normalize_for_duplicate)
    dup = work.duplicated(subset=["_norm_question", "_norm_answer"], keep="first")
    removed_duplicates = int(dup.sum())
    work = work.loc[~dup].copy().drop(columns=["_norm_question", "_norm_answer"])

    work = work.reset_index(drop=True)
    work.insert(0, "parent_id", [f"MEDQ{i:06d}" for i in range(1, len(work)+1)])
    work = work[["parent_id", "question", "answer", "source", "focus_area"]]

    report = {
        "rows_before": int(start),
        "removed_incomplete_question_or_answer": removed_incomplete,
        "removed_empty_after_cleaning": removed_empty_after_clean,
        "removed_duplicates_after_normalization": removed_duplicates,
        "rows_after": int(len(work)),
    }
    return work, report


def save_preprocessed(df: pd.DataFrame, output: Path, report_path: Path | None = None, report: dict | None = None) -> None:
    ensure_parent(output); df.to_csv(output, index=False)
    if report_path and report is not None:
        save_json(report, report_path)
