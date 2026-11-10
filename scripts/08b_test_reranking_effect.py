import _bootstrap  # noqa: F401

import json
import pandas as pd

from src.common import load_config, resolve_path
from src.retrieval import run_all_retrieval


cfg = load_config()
e = cfg["experiment"]

qa = pd.read_csv(
    resolve_path(cfg["paths"]["qa_evaluation_csv"])
)

kb_en = pd.read_csv(
    resolve_path(cfg["paths"]["kb_en_csv"])
)

kb_id = pd.read_csv(
    resolve_path(cfg["paths"]["kb_id_csv"])
)


# =========================================================
# AMBIL 5 PAIR = 10 QUERY (EN + ID)
# =========================================================

pair_ids = (
    qa["pair_id"]
    .drop_duplicates()
    .head(5)
)

test_qa = (
    qa[
        qa["pair_id"].isin(pair_ids)
    ]
    .copy()
)

print(
    f"\nJumlah query test: {len(test_qa)}"
)


test_output = (
    resolve_path(cfg["paths"]["retrieval_dir"])
    / "test_reranking_effect"
)


# =========================================================
# RUN
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

    output_dir=test_output,

    embedding_batch_size=
        e["embedding_batch_size"],

    reranker_batch_size=
        e["reranker_batch_size"],

    reranker_max_length=
        e["reranker_max_length"],
)


# =========================================================
# LOAD RRF BEFORE RERANKING
# =========================================================

before = pd.read_csv(
    test_output /
    "s3_rrf_before_rerank.csv"
)

after = (
    out[
        out["scenario"]
        == "S3_HYBRID_RRF_RERANK"
    ]
    .copy()
)


# =========================================================
# HELPER
# =========================================================

def parse_relevant(value):
    if isinstance(value, list):
        return value
    return json.loads(value)


def first_relevant_rank(
    df,
    relevant_docs,
):
    relevant_docs = set(relevant_docs)

    df = df.sort_values("rank")

    for row in df.itertuples(
        index=False
    ):
        if row.doc_id in relevant_docs:
            return int(row.rank)

    return None


# =========================================================
# COMPARE BEFORE VS AFTER
# =========================================================

rows = []

for q in test_qa.itertuples(index=False):

    relevant = parse_relevant(
        q.relevant_doc_ids
    )

    old = before[
        before["qa_id"] == q.qa_id
    ]

    new = after[
        after["qa_id"] == q.qa_id
    ]

    old_rank = first_relevant_rank(
        old,
        relevant,
    )

    new_rank = first_relevant_rank(
        new,
        relevant,
    )

    if old_rank is None and new_rank is None:
        status = "SAME_NOT_FOUND"

    elif old_rank is None:
        status = "IMPROVED"

    elif new_rank is None:
        status = "WORSE"

    elif new_rank < old_rank:
        status = "IMPROVED"

    elif new_rank > old_rank:
        status = "WORSE"

    else:
        status = "SAME"

    rows.append(
        {
            "qa_id": q.qa_id,
            "pair_id": q.pair_id,
            "language": q.language,
            "question": q.question,
            "rrf_relevant_rank":
                old_rank,
            "rerank_relevant_rank":
                new_rank,
            "status":
                status,
        }
    )


comparison = pd.DataFrame(rows)


print(
    "\n============================================"
)
print(
    "RRF VS BGE RERANKING"
)
print(
    "============================================"
)

print(
    comparison[
        [
            "qa_id",
            "language",
            "rrf_relevant_rank",
            "rerank_relevant_rank",
            "status",
        ]
    ].to_string(index=False)
)


print(
    "\n============================================"
)
print(
    "RINGKASAN"
)
print(
    "============================================"
)

print(
    comparison["status"]
    .value_counts()
    .to_string()
)


comparison.to_csv(
    test_output /
    "reranking_comparison.csv",
    index=False,
)