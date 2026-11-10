from __future__ import annotations
import ast, json
from pathlib import Path
import pandas as pd
from .common import ensure_parent


def create_annotation_candidates(sample_en: pd.DataFrame, kb_en: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    by_parent = {k: g for k,g in kb_en.groupby("parent_id", sort=False)}
    for q in sample_en.itertuples(index=False):
        if q.parent_id not in by_parent:
            raise ValueError(f"Tidak ada chunk untuk parent_id={q.parent_id}")
        for d in by_parent[q.parent_id].itertuples(index=False):
            rows.append({
                "pair_id": q.pair_id,
                "qa_id": f"{q.pair_id}-en",
                "question": q.question,
                "parent_id": q.parent_id,
                "doc_id": d.doc_id,
                "content_id": d.content_id,
                "chunk_text": d.content,
                "is_relevant": "",
                "notes": "",
            })
    return pd.DataFrame(rows)


def save_annotation(df: pd.DataFrame, path: Path):
    ensure_parent(path); df.to_csv(path, index=False)


def _truthy(v) -> bool:
    if pd.isna(v): return False
    return str(v).strip().lower() in {"1","true","yes","y","relevant","ya"}


def build_qa_evaluation(sample_en: pd.DataFrame, qa_id: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    annotation = annotation.copy()
    labels = pd.to_numeric(annotation["is_relevant"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Seluruh baris anotasi harus diisi 0 atau 1 sebelum --apply.")
    if annotation.duplicated(["pair_id", "doc_id"]).any():
        raise ValueError("Duplikasi pasangan pair_id/doc_id pada anotasi.")
    annotation["_rel"] = labels.eq(1)
    rel = annotation[annotation["_rel"]].groupby("pair_id")["doc_id"].apply(list).to_dict()
    missing = [p for p in sample_en["pair_id"] if p not in rel or not rel[p]]
    if missing:
        raise ValueError("Ground truth belum lengkap. QA tanpa relevant chunk: " + ", ".join(missing[:20]))
    id_by_parent = qa_id.set_index("parent_id")
    rows=[]
    for q in sample_en.itertuples(index=False):
        en_docs=rel[q.pair_id]
        id_docs=[d[:-3]+"-id" if d.endswith("-en") else d+"-id" for d in en_docs]
        rows.append({"qa_id":f"{q.pair_id}-en","pair_id":q.pair_id,"language":"en","question":q.question,
                     "reference_answer":q.answer,"relevant_doc_ids":json.dumps(en_docs, ensure_ascii=False),"focus_area":q.focus_area})
        if q.parent_id not in id_by_parent.index:
            raise ValueError(f"Terjemahan QA tidak ditemukan: {q.parent_id}")
        qi=id_by_parent.loc[q.parent_id]
        rows.append({"qa_id":f"{q.pair_id}-id","pair_id":q.pair_id,"language":"id","question":qi["question"],
                     "reference_answer":qi["reference_answer"],"relevant_doc_ids":json.dumps(id_docs, ensure_ascii=False),"focus_area":q.focus_area})
    return pd.DataFrame(rows)


def create_final_translation_review(qa_eval: pd.DataFrame, kb_en: pd.DataFrame, kb_id: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    en_q=qa_eval[qa_eval.language=="en"].set_index("pair_id"); id_q=qa_eval[qa_eval.language=="id"].set_index("pair_id")
    en_docs=kb_en.set_index("content_id"); id_docs=kb_id.set_index("content_id")
    for pair in en_q.index:
        rows += [
            {"pair_id":pair,"type":"question","content_id":"","english_text":en_q.loc[pair,"question"],"indonesian_text":id_q.loc[pair,"question"]},
            {"pair_id":pair,"type":"reference_answer","content_id":"","english_text":en_q.loc[pair,"reference_answer"],"indonesian_text":id_q.loc[pair,"reference_answer"]},
        ]
        doc_ids=json.loads(en_q.loc[pair,"relevant_doc_ids"])
        for doc_id in doc_ids:
            content_id=doc_id[:-3] if doc_id.endswith("-en") else doc_id
            rows.append({"pair_id":pair,"type":"relevant_chunk","content_id":content_id,
                         "english_text":en_docs.loc[content_id,"content"],"indonesian_text":id_docs.loc[content_id,"content"]})
    out=pd.DataFrame(rows); out["is_valid"]=""; out["revision"]=""; out["notes"]=""
    return out
