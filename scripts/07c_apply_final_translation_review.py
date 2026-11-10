import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path(__file__).resolve().parents[1]

REVIEW = ROOT / "data/manual_review/final_evaluation_translation_review.csv"
QA_EVAL = ROOT / "data/evaluation/qa_evaluation.csv"
QA_ID = ROOT / "data/bilingual/qa_pairs_id.csv"
SAMPLE = ROOT / "data/evaluation/qa_sample_en.csv"
KB_ID = ROOT / "data/knowledge_base/knowledge_base_id.csv"
CHUNKS_ID = ROOT / "data/bilingual/chunks_id.csv"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================
# BACKUP
# ============================================================

for path in [REVIEW, QA_EVAL, QA_ID, KB_ID, CHUNKS_ID]:
    if path.exists():
        backup = path.with_name(
            f"{path.stem}_before_final_translation_apply_{timestamp}{path.suffix}"
        )
        shutil.copy2(path, backup)
        print("BACKUP:", backup)


# ============================================================
# LOAD
# ============================================================

review = pd.read_csv(REVIEW)
qa_eval = pd.read_csv(QA_EVAL)
qa_id = pd.read_csv(QA_ID)
sample = pd.read_csv(SAMPLE)
kb_id = pd.read_csv(KB_ID)
chunks_id = pd.read_csv(CHUNKS_ID)


# ============================================================
# VALIDASI REVIEW
# ============================================================

status = pd.to_numeric(
    review["is_valid"],
    errors="coerce"
)

if status.isna().any():
    raise ValueError(
        f"Masih ada {status.isna().sum()} baris review yang belum selesai."
    )

if not (status == 1).all():
    raise ValueError(
        "Ada baris dengan is_valid bukan 1."
    )

revision = (
    review["revision"]
    .fillna("")
    .astype(str)
    .str.strip()
)

review["_final_text"] = review["indonesian_text"].astype(str)

has_revision = revision.ne("")

review.loc[
    has_revision,
    "_final_text"
] = revision[has_revision]

print("\nTOTAL REVIEW    :", len(review))
print("TOTAL PAIR      :", review["pair_id"].nunique())
print("TOTAL REVISION  :", int(has_revision.sum()))


# ============================================================
# CEK DUPLICATE CONTENT_ID DENGAN REVISI BERBEDA
# ============================================================

chunk_review = review[
    review["type"].eq("relevant_chunk")
].copy()

conflicts = []

for content_id, g in chunk_review.groupby("content_id"):

    texts = (
        g["_final_text"]
        .astype(str)
        .str.strip()
        .unique()
    )

    if len(texts) > 1:
        conflicts.append(
            (content_id, texts.tolist())
        )

if conflicts:
    print("\nDITEMUKAN KONFLIK TERJEMAHAN CONTENT_ID:")

    for cid, texts in conflicts:
        print("\n", cid)
        for t in texts:
            print("-", t)

    raise ValueError(
        "Ada content_id yang memiliki lebih dari satu final translation."
    )


# ============================================================
# IDENTIFIKASI ROW INDONESIA PADA QA EVALUATION
# ============================================================

if "language" in qa_eval.columns:

    lang = (
        qa_eval["language"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    id_mask = lang.isin([
        "id",
        "indonesian",
        "indonesia",
        "bahasa indonesia"
    ])

elif "qa_id" in qa_eval.columns:

    qaid = (
        qa_eval["qa_id"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    id_mask = (
        qaid.str.endswith("-id")
        | qaid.str.endswith("_id")
    )

else:
    raise ValueError(
        "Tidak dapat menentukan row Bahasa Indonesia di qa_evaluation.csv."
    )

if id_mask.sum() != 300:
    raise ValueError(
        f"Jumlah query ID di qa_evaluation bukan 300, tetapi {id_mask.sum()}."
    )


# ============================================================
# APPLY QUESTION KE QA_EVALUATION
# ============================================================

question_review = review[
    review["type"].eq("question")
]

if len(question_review) != 300:
    raise ValueError(
        f"Jumlah question review bukan 300: {len(question_review)}"
    )

for _, r in question_review.iterrows():

    mask = (
        qa_eval["pair_id"]
        .astype(str)
        .eq(str(r["pair_id"]))
        & id_mask
    )

    if mask.sum() != 1:
        raise ValueError(
            f"{r['pair_id']}: row ID di qa_evaluation ditemukan {mask.sum()}."
        )

    qa_eval.loc[
        mask,
        "question"
    ] = r["_final_text"]


# ============================================================
# APPLY REFERENCE ANSWER KE QA_EVALUATION
# ============================================================

ref_review = review[
    review["type"].eq("reference_answer")
]

if len(ref_review) != 300:
    raise ValueError(
        f"Jumlah reference answer review bukan 300: {len(ref_review)}"
    )

if "reference_answer" in qa_eval.columns:
    eval_ref_col = "reference_answer"
elif "answer" in qa_eval.columns:
    eval_ref_col = "answer"
else:
    raise ValueError(
        "Kolom reference_answer/answer tidak ditemukan di qa_evaluation."
    )

for _, r in ref_review.iterrows():

    mask = (
        qa_eval["pair_id"]
        .astype(str)
        .eq(str(r["pair_id"]))
        & id_mask
    )

    if mask.sum() != 1:
        raise ValueError(
            f"{r['pair_id']}: reference row ID ditemukan {mask.sum()}."
        )

    qa_eval.loc[
        mask,
        eval_ref_col
    ] = r["_final_text"]


# ============================================================
# APPLY QUESTION + ANSWER KE QA_PAIRS_ID
# BERDASARKAN PARENT_ID DARI SAMPLE
# ============================================================

pair_to_parent = dict(
    zip(
        sample["pair_id"].astype(str),
        sample["parent_id"].astype(str)
    )
)

if "parent_id" not in qa_id.columns:
    raise ValueError(
        "qa_pairs_id.csv tidak memiliki parent_id."
    )

qa_id["parent_id"] = (
    qa_id["parent_id"]
    .astype(str)
    .str.strip()
)

for _, r in question_review.iterrows():

    pair_id = str(r["pair_id"])

    if pair_id not in pair_to_parent:
        raise ValueError(
            f"{pair_id} tidak ditemukan di qa_sample_en.csv."
        )

    parent_id = pair_to_parent[pair_id]

    mask = qa_id["parent_id"].eq(parent_id)

    if mask.sum() != 1:
        raise ValueError(
            f"{parent_id}: ditemukan {mask.sum()} row di qa_pairs_id.csv."
        )

    qa_id.loc[
        mask,
        "question"
    ] = r["_final_text"]


# Tentukan kolom reference answer pada QA Indonesia
if "reference_answer" in qa_id.columns:
    qa_id_ref_col = "reference_answer"
elif "answer" in qa_id.columns:
    qa_id_ref_col = "answer"
else:
    raise ValueError(
        "qa_pairs_id.csv tidak mempunyai kolom reference_answer/answer."
    )

for _, r in ref_review.iterrows():

    pair_id = str(r["pair_id"])
    parent_id = pair_to_parent[pair_id]

    mask = qa_id["parent_id"].eq(parent_id)

    if mask.sum() != 1:
        raise ValueError(
            f"{parent_id}: ditemukan {mask.sum()} row untuk reference answer."
        )

    qa_id.loc[
        mask,
        qa_id_ref_col
    ] = r["_final_text"]


# ============================================================
# APPLY RELEVANT CHUNK
# KE KB_ID + CHUNKS_ID
# ============================================================

unique_chunks = (
    chunk_review[
        ["content_id", "_final_text"]
    ]
    .drop_duplicates(
        subset=["content_id"]
    )
)

print(
    "RELEVANT CHUNK  :",
    len(chunk_review)
)

print(
    "UNIQUE CONTENT  :",
    len(unique_chunks)
)

for _, r in unique_chunks.iterrows():

    cid = str(r["content_id"]).strip()
    final_text = r["_final_text"]

    # KB
    mask_kb = (
        kb_id["content_id"]
        .astype(str)
        .str.strip()
        .eq(cid)
    )

    if mask_kb.sum() != 1:
        raise ValueError(
            f"{cid}: ditemukan {mask_kb.sum()} row di knowledge_base_id.csv."
        )

    kb_id.loc[
        mask_kb,
        "content"
    ] = final_text

    # chunks_id
    mask_chunks = (
        chunks_id["content_id"]
        .astype(str)
        .str.strip()
        .eq(cid)
    )

    if mask_chunks.sum() != 1:
        raise ValueError(
            f"{cid}: ditemukan {mask_chunks.sum()} row di chunks_id.csv."
        )

    chunks_id.loc[
        mask_chunks,
        "content"
    ] = final_text


# ============================================================
# SAVE
# ============================================================

qa_eval.to_csv(
    QA_EVAL,
    index=False
)

qa_id.to_csv(
    QA_ID,
    index=False
)

kb_id.to_csv(
    KB_ID,
    index=False
)

chunks_id.to_csv(
    CHUNKS_ID,
    index=False
)


# ============================================================
# VALIDASI FINAL
# ============================================================

qa_eval_check = pd.read_csv(QA_EVAL)
qa_id_check = pd.read_csv(QA_ID)
kb_check = pd.read_csv(KB_ID)
chunks_check = pd.read_csv(CHUNKS_ID)

if "language" in qa_eval_check.columns:

    langs = (
        qa_eval_check["language"]
        .astype(str)
        .str.lower()
        .str.strip()
        .value_counts()
    )

    print("\nLANGUAGE DISTRIBUTION:")
    print(langs)

print("\n" + "=" * 70)
print("VALIDASI APPLY FINAL TRANSLATION")
print("=" * 70)

question_mismatch = 0
reference_mismatch = 0
kb_mismatch = 0
chunks_mismatch = 0

# ID mask baru
if "language" in qa_eval_check.columns:
    lang = (
        qa_eval_check["language"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    id_mask_check = lang.isin([
        "id",
        "indonesian",
        "indonesia",
        "bahasa indonesia"
    ])
else:
    qaid = (
        qa_eval_check["qa_id"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    id_mask_check = (
        qaid.str.endswith("-id")
        | qaid.str.endswith("_id")
    )


# question
for _, r in question_review.iterrows():

    row = qa_eval_check[
        qa_eval_check["pair_id"]
        .astype(str)
        .eq(str(r["pair_id"]))
        & id_mask_check
    ]

    if (
        len(row) != 1
        or str(row.iloc[0]["question"]).strip()
        != str(r["_final_text"]).strip()
    ):
        question_mismatch += 1


# reference
for _, r in ref_review.iterrows():

    row = qa_eval_check[
        qa_eval_check["pair_id"]
        .astype(str)
        .eq(str(r["pair_id"]))
        & id_mask_check
    ]

    if (
        len(row) != 1
        or str(row.iloc[0][eval_ref_col]).strip()
        != str(r["_final_text"]).strip()
    ):
        reference_mismatch += 1


# chunks
for _, r in unique_chunks.iterrows():

    cid = str(r["content_id"]).strip()
    expected = str(r["_final_text"]).strip()

    k = kb_check[
        kb_check["content_id"]
        .astype(str)
        .str.strip()
        .eq(cid)
    ]

    c = chunks_check[
        chunks_check["content_id"]
        .astype(str)
        .str.strip()
        .eq(cid)
    ]

    if (
        len(k) != 1
        or str(k.iloc[0]["content"]).strip() != expected
    ):
        kb_mismatch += 1

    if (
        len(c) != 1
        or str(c.iloc[0]["content"]).strip() != expected
    ):
        chunks_mismatch += 1


print("TOTAL QA EVALUATION     :", len(qa_eval_check))
print(
    "TOTAL PAIR EVALUATION   :",
    qa_eval_check["pair_id"].nunique()
)
print(
    "QUERY ID                :",
    int(id_mask_check.sum())
)
print(
    "QUERY NON-ID            :",
    int((~id_mask_check).sum())
)

print(
    "QUESTION MISMATCH       :",
    question_mismatch
)

print(
    "REFERENCE MISMATCH      :",
    reference_mismatch
)

print(
    "KB CHUNK MISMATCH       :",
    kb_mismatch
)

print(
    "SOURCE CHUNK MISMATCH   :",
    chunks_mismatch
)

print(
    "REVISION DITERAPKAN     :",
    int(has_revision.sum())
)

if (
    len(qa_eval_check) == 600
    and qa_eval_check["pair_id"].nunique() == 300
    and id_mask_check.sum() == 300
    and question_mismatch == 0
    and reference_mismatch == 0
    and kb_mismatch == 0
    and chunks_mismatch == 0
):
    print("\n" + "=" * 70)
    print("SUKSES")
    print("=" * 70)
    print(
        "Semua final translation sudah diterapkan dan sinkron."
    )
else:
    raise ValueError(
        "Validasi final gagal. Periksa nilai mismatch di atas."
    )
