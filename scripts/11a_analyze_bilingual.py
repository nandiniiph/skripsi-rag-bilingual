import _bootstrap  # noqa: F401

import json
import pandas as pd

from src.common import load_config, resolve_path


cfg = load_config()


# =========================================================
# LOAD DATA
# =========================================================

qa = pd.read_csv(
    resolve_path(
        cfg["paths"]["qa_evaluation_csv"]
    )
)

retrieval = pd.read_csv(
    resolve_path(
        cfg["paths"]["retrieval_dir"]
    ) / "retrieval_results.csv"
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

generation = pd.read_csv(
    resolve_path(
        cfg["paths"]["generation_per_query_csv"]
    )
)


# =========================================================
# DOC_ID → CONTENT_ID
# =========================================================

doc_to_content = {}

for kb in [kb_en, kb_id]:
    for row in kb.itertuples(index=False):
        doc_to_content[row.doc_id] = row.content_id


# =========================================================
# HELPER
# =========================================================

def parse_ids(value):
    if isinstance(value, list):
        return value
    return json.loads(value)


def get_top_docs(
    qa_id,
    scenario,
    k=3,
):
    g = retrieval[
        (retrieval["qa_id"] == qa_id)
        &
        (retrieval["scenario"] == scenario)
    ]

    return (
        g.sort_values("rank")
        .head(k)["doc_id"]
        .tolist()
    )


def to_content_ids(doc_ids):
    return [
        doc_to_content[d]
        for d in doc_ids
        if d in doc_to_content
    ]


def hit_at_k(
    qa_row,
    scenario,
    k,
):
    relevant = set(
        parse_ids(
            qa_row["relevant_doc_ids"]
        )
    )

    retrieved = get_top_docs(
        qa_row["qa_id"],
        scenario,
        k,
    )

    return int(
        len(
            set(retrieved) & relevant
        ) > 0
    )


# =========================================================
# GENERATION LOOKUP
# =========================================================

generation_lookup = generation.set_index(
    [
        "qa_id",
        "scenario",
    ]
)


# =========================================================
# ANALISIS PER PAIR
# =========================================================

scenarios = [
    "S1_BM25",
    "S2_DENSE",
    "S3_HYBRID_RRF_RERANK",
]

rows = []


for pair_id, pair_group in qa.groupby("pair_id"):

    en_rows = pair_group[
        pair_group["language"] == "en"
    ]

    id_rows = pair_group[
        pair_group["language"] == "id"
    ]

    # Pastikan pair lengkap
    if len(en_rows) != 1 or len(id_rows) != 1:
        continue

    en = en_rows.iloc[0]
    ind = id_rows.iloc[0]

    for scenario in scenarios:

        # =============================================
        # TOP-3 RETRIEVAL
        # =============================================

        en_docs = get_top_docs(
            en["qa_id"],
            scenario,
            3,
        )

        id_docs = get_top_docs(
            ind["qa_id"],
            scenario,
            3,
        )

        en_content = set(
            to_content_ids(en_docs)
        )

        id_content = set(
            to_content_ids(id_docs)
        )

        intersection = (
            en_content & id_content
        )

        overlap_at_3 = (
            len(intersection) / 3
        )

        # =============================================
        # HIT
        # =============================================

        en_hit1 = hit_at_k(
            en,
            scenario,
            1,
        )

        id_hit1 = hit_at_k(
            ind,
            scenario,
            1,
        )

        en_hit3 = hit_at_k(
            en,
            scenario,
            3,
        )

        id_hit3 = hit_at_k(
            ind,
            scenario,
            3,
        )

        # =============================================
        # STATUS PAIR TOP-3
        # =============================================

        if en_hit3 and id_hit3:
            pair_status = "BOTH_HIT"

        elif en_hit3 and not id_hit3:
            pair_status = "EN_ONLY"

        elif not en_hit3 and id_hit3:
            pair_status = "ID_ONLY"

        else:
            pair_status = "NEITHER"

        # =============================================
        # GENERATION METRICS
        # =============================================

        en_gen = generation_lookup.loc[
            (
                en["qa_id"],
                scenario,
            )
        ]

        id_gen = generation_lookup.loc[
            (
                ind["qa_id"],
                scenario,
            )
        ]

        rows.append(
            {
                "pair_id":
                    pair_id,

                "scenario":
                    scenario,

                "qa_id_en":
                    en["qa_id"],

                "qa_id_id":
                    ind["qa_id"],

                "en_hit_at_1":
                    en_hit1,

                "id_hit_at_1":
                    id_hit1,

                "en_hit_at_3":
                    en_hit3,

                "id_hit_at_3":
                    id_hit3,

                "pair_status_at_3":
                    pair_status,

                "top3_content_overlap":
                    overlap_at_3,

                "en_rougeL_f1":
                    en_gen["rougeL_f1"],

                "id_rougeL_f1":
                    id_gen["rougeL_f1"],

                "rougeL_abs_difference":
                    abs(
                        en_gen["rougeL_f1"]
                        -
                        id_gen["rougeL_f1"]
                    ),

                "en_bertscore_f1":
                    en_gen["bertscore_f1"],

                "id_bertscore_f1":
                    id_gen["bertscore_f1"],

                "bertscore_abs_difference":
                    abs(
                        en_gen["bertscore_f1"]
                        -
                        id_gen["bertscore_f1"]
                    ),
            }
        )


pair_analysis = pd.DataFrame(rows)


# =========================================================
# SUMMARY
# =========================================================

summary_rows = []


for scenario in scenarios:

    subset = pair_analysis[
        pair_analysis["scenario"]
        == scenario
    ]

    counts = (
        subset[
            "pair_status_at_3"
        ]
        .value_counts()
        .to_dict()
    )

    summary_rows.append(
        {
            "scenario":
                scenario,

            "pairs":
                len(subset),

            "both_hit_at_3":
                counts.get(
                    "BOTH_HIT",
                    0,
                ),

            "en_only_hit_at_3":
                counts.get(
                    "EN_ONLY",
                    0,
                ),

            "id_only_hit_at_3":
                counts.get(
                    "ID_ONLY",
                    0,
                ),

            "neither_hit_at_3":
                counts.get(
                    "NEITHER",
                    0,
                ),

            "mean_top3_content_overlap":
                subset[
                    "top3_content_overlap"
                ].mean(),

            "mean_en_rougeL_f1":
                subset[
                    "en_rougeL_f1"
                ].mean(),

            "mean_id_rougeL_f1":
                subset[
                    "id_rougeL_f1"
                ].mean(),

            "mean_rougeL_abs_difference":
                subset[
                    "rougeL_abs_difference"
                ].mean(),

            "mean_en_bertscore_f1":
                subset[
                    "en_bertscore_f1"
                ].mean(),

            "mean_id_bertscore_f1":
                subset[
                    "id_bertscore_f1"
                ].mean(),

            "mean_bertscore_abs_difference":
                subset[
                    "bertscore_abs_difference"
                ].mean(),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# =========================================================
# PRINT
# =========================================================

print(
    "\n=================================================="
)
print(
    "BILINGUAL PAIR ANALYSIS"
)
print(
    "=================================================="
)

print(
    summary.to_string(
        index=False
    )
)


print(
    "\n=================================================="
)
print(
    "PAIR STATUS @3"
)
print(
    "=================================================="
)

status_table = (
    pair_analysis
    .groupby(
        [
            "scenario",
            "pair_status_at_3",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)

print(
    status_table.to_string(
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
            "final_results_csv"
        ]
    ).parent
)

evaluation_dir.mkdir(
    parents=True,
    exist_ok=True,
)


pair_analysis.to_csv(
    evaluation_dir /
    "bilingual_pair_analysis.csv",
    index=False,
)

summary.to_csv(
    evaluation_dir /
    "bilingual_summary.csv",
    index=False,
)


print(
    "\nFiles saved:"
)

print(
    evaluation_dir /
    "bilingual_pair_analysis.csv"
)

print(
    evaluation_dir /
    "bilingual_summary.csv"
)