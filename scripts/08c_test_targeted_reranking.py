import _bootstrap  # noqa: F401

import json
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

# Hasil RRF lama
old_retrieval = pd.read_csv(
    "../skripsi/outputs/retrieval/retrieval_results.csv"
)

old_rrf = old_retrieval[
    old_retrieval["scenario"] == "S3_HYBRID_RRF"
].copy()


# =========================================================
# HELPER
# =========================================================

def parse_relevant(value):
    if isinstance(value, list):
        return value
    return json.loads(value)


gt = (
    qa.set_index("qa_id")["relevant_doc_ids"]
    .map(parse_relevant)
    .to_dict()
)


def first_relevant_rank(
    df,
    relevant_docs,
):
    relevant_docs = set(relevant_docs)

    df = df.sort_values("rank")

    for row in df.itertuples(index=False):
        if row.doc_id in relevant_docs:
            return int(row.rank)

    return None


# =========================================================
# CARI QUERY DENGAN RELEVANT DOC RANK 2-10
# =========================================================

candidate_rows = []

for qa_id, group in old_rrf.groupby("qa_id"):

    relevant = gt[qa_id]

    rank = first_relevant_rank(
        group,
        relevant,
    )

    if rank is None:
        continue

    if 2 <= rank <= 10:

        q = qa[
            qa["qa_id"] == qa_id
        ].iloc[0]

        candidate_rows.append(
            {
                "qa_id": qa_id,
                "language": q["language"],
                "old_rrf_rank": rank,
            }
        )


candidates = pd.DataFrame(
    candidate_rows
)


# =========================================================
# SAMPLE 10 EN + 10 ID
# =========================================================

sample_en = (
    candidates[
        candidates["language"] == "en"
    ]
    .sample(
        n=min(
            10,
            len(
                candidates[
                    candidates["language"]
                    == "en"
                ]
            ),
        ),
        random_state=42,
    )
)


sample_id = (
    candidates[
        candidates["language"] == "id"
    ]
    .sample(
        n=min(
            10,
            len(
                candidates[
                    candidates["language"]
                    == "id"
                ]
            ),
        ),
        random_state=42,
    )
)


selected = pd.concat(
    [
        sample_en,
        sample_id,
    ],
    ignore_index=True,
)


test_qa = qa[
    qa["qa_id"].isin(
        selected["qa_id"]
    )
].copy()


print(
    "\n========================================"
)
print(
    "TARGETED TEST QUERY"
)
print(
    "========================================"
)

print(
    selected
    .sort_values(
        [
            "language",
            "old_rrf_rank",
        ]
    )
    .to_string(index=False)
)


# =========================================================
# OUTPUT
# =========================================================

test_output = (
    resolve_path(
        cfg["paths"]["retrieval_dir"]
    )
    / "test_targeted_reranking"
)


# =========================================================
# RUN NEW PIPELINE
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
# BEFORE RERANK
# =========================================================

before = pd.read_csv(
    test_output /
    "s3_rrf_before_rerank.csv"
)


# =========================================================
# AFTER RERANK
# =========================================================

after = (
    out[
        out["scenario"]
        == "S3_HYBRID_RRF_RERANK"
    ]
    .copy()
)


# =========================================================
# COMPARISON
# =========================================================

rows = []

for q in test_qa.itertuples(index=False):

    relevant = parse_relevant(
        q.relevant_doc_ids
    )

    old_group = before[
        before["qa_id"] == q.qa_id
    ]

    new_group = after[
        after["qa_id"] == q.qa_id
    ]

    old_rank = first_relevant_rank(
        old_group,
        relevant,
    )

    new_rank = first_relevant_rank(
        new_group,
        relevant,
    )

    if (
        old_rank is not None
        and new_rank is not None
    ):
        if new_rank < old_rank:
            status = "IMPROVED"

        elif new_rank > old_rank:
            status = "WORSE"

        else:
            status = "SAME"

    else:
        status = "NOT_FOUND"

    rows.append(
        {
            "qa_id":
                q.qa_id,

            "language":
                q.language,

            "rrf_rank":
                old_rank,

            "reranked_rank":
                new_rank,

            "rank_change":
                (
                    old_rank - new_rank
                    if old_rank is not None
                    and new_rank is not None
                    else None
                ),

            "status":
                status,
        }
    )


comparison = pd.DataFrame(rows)


# =========================================================
# PRINT
# =========================================================

print(
    "\n========================================"
)
print(
    "RRF VS BGE RERANKER"
)
print(
    "========================================"
)

print(
    comparison
    .sort_values(
        [
            "language",
            "qa_id",
        ]
    )
    .to_string(index=False)
)


print(
    "\n========================================"
)
print(
    "STATUS"
)
print(
    "========================================"
)

print(
    comparison[
        "status"
    ]
    .value_counts()
    .to_string()
)


print(
    "\n========================================"
)
print(
    "RATA-RATA RANK RELEVANT"
)
print(
    "========================================"
)

print(
    "RRF before :",
    comparison[
        "rrf_rank"
    ].mean()
)

print(
    "After BGE  :",
    comparison[
        "reranked_rank"
    ].mean()
)


comparison.to_csv(
    test_output /
    "targeted_reranking_comparison.csv",
    index=False,
)