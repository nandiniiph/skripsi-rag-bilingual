from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from .common import ensure_parent, save_json

REQUIRED_COLUMNS = ["question", "answer", "source", "focus_area"]

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
HTML_RE = re.compile(r"<[^>]+>")
NAV_HINT_RE = re.compile(
    r"(?i)(?:watch (?:the|this|an?) (?:animated )?video|click here|see this graphic|see a glossary|"
    r"to enlarge the video|press the escape|read more|learn more)"
)


def _word_len(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split().str.len()


def analyze_dataset(df: pd.DataFrame, analysis_json: Path, focus_csv: Path, artifact_csv: Path) -> dict:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing_cols}. Ditemukan: {list(df.columns)}")

    q = df["question"].fillna("").astype(str)
    a = df["answer"].fillna("").astype(str)
    qw, aw = _word_len(q), _word_len(a)

    exact_dup = df.fillna("").duplicated(subset=["question", "answer"], keep="first")
    artifact_mask = a.str.contains(URL_RE) | a.str.contains(HTML_RE) | a.str.contains(NAV_HINT_RE)

    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "missing_question": int(q.str.strip().eq("").sum()),
        "missing_answer": int(a.str.strip().eq("").sum()),
        "missing_source": int(df["source"].isna().sum()),
        "missing_focus_area": int(df["focus_area"].isna().sum()),
        "exact_duplicate_question_answer": int(exact_dup.sum()),
        "unique_source": int(df["source"].nunique(dropna=True)),
        "unique_focus_area": int(df["focus_area"].nunique(dropna=True)),
        "question_words": {"min": int(qw.min()), "mean": float(qw.mean()), "median": float(qw.median()), "max": int(qw.max())},
        "answer_words": {"min": int(aw.min()), "mean": float(aw.mean()), "median": float(aw.median()), "max": int(aw.max())},
        "rows_with_url_html_or_navigation_hint": int(artifact_mask.sum()),
    }
    save_json(report, analysis_json)

    focus = (df["focus_area"].fillna("[MISSING]").value_counts(dropna=False)
             .rename_axis("focus_area").reset_index(name="count"))
    ensure_parent(focus_csv); focus.to_csv(focus_csv, index=False)

    artifacts = df.loc[artifact_mask, REQUIRED_COLUMNS].copy()
    artifacts.insert(0, "raw_row", artifacts.index)
    ensure_parent(artifact_csv); artifacts.to_csv(artifact_csv, index=False)
    return report
