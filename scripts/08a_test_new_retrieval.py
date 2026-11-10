import _bootstrap  # noqa: F401

import pandas as pd

from src.common import load_config, resolve_path
from src.retrieval import run_all_retrieval


cfg = load_config()
e = cfg["experiment"]


# =========================================================
# LOAD DATA
# =========================================================

qa = pd.read_csv(
    resolve_path(
        cfg["paths"]["qa_evaluation_csv"]
    )
)

kb_en = pd.read_csv(
    resolve_path(
        cfg["paths"]["kb_en_csv"]
    )
)

kb_id = pd.read_csv(
    resolve_path(
        cfg["paths"]["kb_id_csv"]
    )
)


# =========================================================
# AMBIL 1 QUERY ENGLISH
# =========================================================

test_qa = (
    qa[
        qa["language"] == "en"
    ]
    .head(1)
    .copy()
)


print("\n========================================")
print("QUERY TEST")
print("========================================")

print(
    test_qa[
        [
            "qa_id",
            "pair_id",
            "language",
            "question",
        ]
    ].to_string(index=False)
)


# =========================================================
# OUTPUT TEST DIPISAHKAN
# =========================================================

test_output = (
    resolve_path(
        cfg["paths"]["retrieval_dir"]
    )
    / "test_new"
)


# =========================================================
# RUN RETRIEVAL
# =========================================================

out = run_all_retrieval(
    qa_eval=test_qa,

    kb_en=kb_en,
    kb_id=kb_id,

    embedding_model=
        cfg["models"]["embedding"],

    reranker_model=
        cfg["models"]["reranker"],

    bm25_top_n=
        e["bm25_top_n"],

    dense_top_n=
        e["dense_top_n"],

    rrf_k=
        e["rrf_k"],

    rrf_candidate_top_n=
        e["rrf_candidate_top_n"],

    output_dir=
        test_output,

    embedding_batch_size=
        e["embedding_batch_size"],

    reranker_batch_size=
        e["reranker_batch_size"],

    reranker_max_length=
        e["reranker_max_length"],
)


# =========================================================
# TAMPILKAN S1
# =========================================================

print("\n========================================")
print("S1 BM25 TOP-5")
print("========================================")

s1 = (
    out[
        out["scenario"]
        == "S1_BM25"
    ]
    .sort_values("rank")
    .head(5)
)

print(
    s1[
        [
            "doc_id",
            "rank",
            "score",
        ]
    ].to_string(index=False)
)


# =========================================================
# TAMPILKAN S2
# =========================================================

print("\n========================================")
print("S2 DENSE TOP-5")
print("========================================")

s2 = (
    out[
        out["scenario"]
        == "S2_DENSE"
    ]
    .sort_values("rank")
    .head(5)
)

print(
    s2[
        [
            "doc_id",
            "rank",
            "score",
        ]
    ].to_string(index=False)
)


# =========================================================
# TAMPILKAN S3
# =========================================================

print("\n========================================")
print("S3 HYBRID RRF + BGE TOP-10")
print("========================================")

s3 = (
    out[
        out["scenario"]
        == "S3_HYBRID_RRF_RERANK"
    ]
    .sort_values("rank")
    .head(10)
)

print(
    s3[
        [
            "doc_id",
            "rank",
            "rrf_rank",
            "rrf_score",
            "reranker_score",
        ]
    ].to_string(index=False)
)


# =========================================================
# GROUND TRUTH
# =========================================================

print("\n========================================")
print("GROUND TRUTH")
print("========================================")

print(
    test_qa[
        [
            "qa_id",
            "relevant_doc_ids",
        ]
    ].to_string(index=False)
)