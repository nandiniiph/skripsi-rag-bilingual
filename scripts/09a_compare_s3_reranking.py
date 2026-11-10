import _bootstrap  # noqa: F401

import json
import pandas as pd

from src.common import load_config, resolve_path
from src.metrics import evaluate_retrieval


cfg = load_config()

# =========================================================
# LOAD DATA
# =========================================================

qa = pd.read_csv(
    resolve_path(
        cfg["paths"]["qa_evaluation_csv"]
    )
)

retrieval_dir = resolve_path(
    cfg["paths"]["retrieval_dir"]
)

before = pd.read_csv(
    retrieval_dir /
    "s3_rrf_before_rerank.csv"
)

all_results = pd.read_csv(
    retrieval_dir /
    "retrieval_results.csv"
)

after = all_results[
    all_results["scenario"]
    == "S3_HYBRID_RRF_RERANK"
].copy()


# =========================================================
# BERIKAN NAMA SKENARIO UNTUK BEFORE
# =========================================================

before["scenario"] = "S3_RRF_BEFORE_RERANK"

# score diperlukan oleh struktur retrieval,
# tetapi tidak digunakan oleh metric evaluator.
before["score"] = before["rrf_score"]


# =========================================================
# SAMAKAN KOLOM
# =========================================================

columns = [
    "qa_id",
    "pair_id",
    "language",
    "predicted_language",
    "scenario",
    "doc_id",
    "rank",
    "score",
]

before_eval = before[columns].copy()
after_eval = after[columns].copy()

comparison_retrieval = pd.concat(
    [
        before_eval,
        after_eval,
    ],
    ignore_index=True,
)


# =========================================================
# EVALUASI METRIK
# =========================================================

per_query, summary = evaluate_retrieval(
    qa,
    comparison_retrieval,
)


print(
    "\n=============================================="
)
print(
    "S3 SEBELUM VS SESUDAH BGE RERANKING"
)
print(
    "=============================================="
)

print(
    summary.to_string(
        index=False
    )
)


# =========================================================
# ANALISIS PERUBAHAN RANK DOKUMEN RELEVAN
# =========================================================

def parse_ids(value):
    if isinstance(value, list):
        return value
    return json.loads(value)


ground_truth = (
    qa.set_index("qa_id")[
        "relevant_doc_ids"
    ]
    .map(parse_ids)
    .to_dict()
)


def first_relevant_rank(
    df,
    relevant_docs,
):
    relevant_docs = set(
        relevant_docs
    )

    df = df.sort_values("rank")

    for row in df.itertuples(
        index=False
    ):
        if row.doc_id in relevant_docs:
            return int(row.rank)

    return None


rows = []

for q in qa.itertuples(index=False):

    relevant = ground_truth[
        q.qa_id
    ]

    old_group = before[
        before["qa_id"]
        == q.qa_id
    ]

    new_group = after[
        after["qa_id"]
        == q.qa_id
    ]

    old_rank = first_relevant_rank(
        old_group,
        relevant,
    )

    new_rank = first_relevant_rank(
        new_group,
        relevant,
    )

    # -----------------------------------------
    # Status perubahan
    # -----------------------------------------

    if (
        old_rank is None
        and new_rank is None
    ):
        status = "NOT_FOUND"

    elif (
        old_rank is None
        and new_rank is not None
    ):
        status = "IMPROVED"

    elif (
        old_rank is not None
        and new_rank is None
    ):
        status = "WORSE"

    elif new_rank < old_rank:
        status = "IMPROVED"

    elif new_rank > old_rank:
        status = "WORSE"

    else:
        status = "SAME"

    if (
        old_rank is not None
        and new_rank is not None
    ):
        rank_change = (
            old_rank - new_rank
        )
    else:
        rank_change = None

    rows.append(
        {
            "qa_id": q.qa_id,
            "pair_id": q.pair_id,
            "language": q.language,
            "question": q.question,

            "rrf_rank":
                old_rank,

            "reranked_rank":
                new_rank,

            "rank_change":
                rank_change,

            "status":
                status,
        }
    )


rank_comparison = pd.DataFrame(
    rows
)


# =========================================================
# RINGKAS STATUS
# =========================================================

print(
    "\n=============================================="
)
print(
    "PERUBAHAN POSISI RELEVANT DOCUMENT"
)
print(
    "=============================================="
)

status_summary = (
    rank_comparison
    .groupby(
        [
            "language",
            "status",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

print(
    status_summary.to_string(
        index=False
    )
)


# =========================================================
# RATA-RATA RANK
# HANYA QUERY YANG RELEVANT DOC-NYA DITEMUKAN
# =========================================================

print(
    "\n=============================================="
)
print(
    "RATA-RATA POSISI RELEVANT DOCUMENT"
)
print(
    "=============================================="
)

for language in ["en", "id"]:

    subset = rank_comparison[
        rank_comparison["language"]
        == language
    ]

    valid = subset[
        subset["rrf_rank"].notna()
        &
        subset["reranked_rank"].notna()
    ]

    print(
        f"\nLanguage: {language}"
    )

    print(
        "RRF before :",
        valid["rrf_rank"].mean()
    )

    print(
        "After BGE  :",
        valid[
            "reranked_rank"
        ].mean()
    )


# =========================================================
# TOP IMPROVEMENTS
# =========================================================

print(
    "\n=============================================="
)
print(
    "10 PERBAIKAN RANK TERBESAR"
)
print(
    "=============================================="
)

top_improved = (
    rank_comparison[
        rank_comparison["status"]
        == "IMPROVED"
    ]
    .sort_values(
        "rank_change",
        ascending=False,
    )
    .head(10)
)

print(
    top_improved[
        [
            "qa_id",
            "language",
            "rrf_rank",
            "reranked_rank",
            "rank_change",
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# TOP DEGRADATIONS
# =========================================================

print(
    "\n=============================================="
)
print(
    "10 PENURUNAN RANK TERBESAR"
)
print(
    "=============================================="
)

top_worse = (
    rank_comparison[
        rank_comparison["status"]
        == "WORSE"
    ]
    .sort_values(
        "rank_change",
        ascending=True,
    )
    .head(10)
)

print(
    top_worse[
        [
            "qa_id",
            "language",
            "rrf_rank",
            "reranked_rank",
            "rank_change",
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# SAVE
# =========================================================

evaluation_dir = (
    resolve_path(
        cfg[
            "paths"
        ][
            "retrieval_summary_csv"
        ]
    ).parent
)

evaluation_dir.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    evaluation_dir /
    "s3_reranking_comparison_summary.csv",
    index=False,
)

per_query.to_csv(
    evaluation_dir /
    "s3_reranking_comparison_per_query.csv",
    index=False,
)

rank_comparison.to_csv(
    evaluation_dir /
    "s3_reranking_rank_changes.csv",
    index=False,
)


print(
    "\nFile analisis tersimpan di:"
)

print(
    evaluation_dir /
    "s3_reranking_comparison_summary.csv"
)

print(
    evaluation_dir /
    "s3_reranking_rank_changes.csv"
)