from __future__ import annotations

import re
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
import pandas as pd
import torch

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.common import (
    load_config,
    resolve_path,
    seed_everything,
)

from src.generation import SailorGenerator
from src.reranking import BGEReranker
from src.language import detect_language


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent
INDEX_DIR = resolve_path(load_config()["paths"]["index_dir"])

cfg = load_config()

seed_everything(
    cfg["project"]["seed"]
)

EMBEDDING_MODEL = cfg["models"]["embedding"]
RERANKER_MODEL = cfg["models"]["reranker"]
GENERATOR_MODEL = cfg["models"]["generator"]

BM25_TOP_N = cfg["experiment"]["bm25_top_n"]
DENSE_TOP_N = cfg["experiment"]["dense_top_n"]

RRF_K = cfg["experiment"]["rrf_k"]

RRF_CANDIDATE_TOP_N = (
    cfg["experiment"]["rrf_candidate_top_n"]
)

TOP_K_EVIDENCE = (
    cfg["experiment"]["top_k_evidence"]
)

MAX_NEW_TOKENS = (
    cfg["experiment"]["max_new_tokens"]
)

RERANKER_BATCH_SIZE = (
    cfg["experiment"]["reranker_batch_size"]
)

RERANKER_MAX_LENGTH = (
    cfg["experiment"]["reranker_max_length"]
)


# ============================================================
# TOKENIZER BM25
# ============================================================

TOKEN_RE = re.compile(
    r"\w+",
    re.UNICODE,
)


def lexical_tokens(text: str):
    return TOKEN_RE.findall(
        str(text).casefold()
    )


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

print("=" * 70)
print("LOADING KNOWLEDGE BASE")
print("=" * 70)

kb = {
    "en": pd.read_csv(
        resolve_path(
            cfg["paths"]["kb_en_csv"]
        )
    ).reset_index(drop=True),

    "id": pd.read_csv(
        resolve_path(
            cfg["paths"]["kb_id_csv"]
        )
    ).reset_index(drop=True),
}

print("KB EN:", len(kb["en"]))
print("KB ID:", len(kb["id"]))


# Lookup doc_id -> content
doc_text = {
    "en": (
        kb["en"]
        .set_index("doc_id")["content"]
        .astype(str)
        .to_dict()
    ),

    "id": (
        kb["id"]
        .set_index("doc_id")["content"]
        .astype(str)
        .to_dict()
    ),
}


# ============================================================
# LOAD BM25
# ============================================================

print("\nBuilding BM25...")

bm25 = {}

for lang in ["en", "id"]:

    tokens = [
        lexical_tokens(x)
        for x in kb[lang]["content"]
    ]

    bm25[lang] = BM25Okapi(
        tokens, k1=1.5, b=0.75, epsilon=0.25
    )

print("BM25 ready.")


# ============================================================
# LOAD E5
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nLoading embedding model...")
print("Device:", device)

encoder = SentenceTransformer(
    EMBEDDING_MODEL,
    device=device,
)

encoder.max_seq_length = 512
print("E5 ready.")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("\nLoading FAISS indexes...")

dense_index = {
    "en": faiss.read_index(
        str(
            INDEX_DIR
            / "multilingual_e5_en.faiss"
        )
    ),

    "id": faiss.read_index(
        str(
            INDEX_DIR
            / "multilingual_e5_id.faiss"
        )
    ),
}

import hashlib, json
for _lang in ["en", "id"]:
    _map = pd.read_csv(INDEX_DIR / f"multilingual_e5_{_lang}_mapping.csv")
    _meta = json.loads((INDEX_DIR / f"multilingual_e5_{_lang}_meta.json").read_text())
    _hash = hashlib.sha256(pd.util.hash_pandas_object(kb[_lang], index=False).values.tobytes()).hexdigest()
    if _map.doc_id.tolist() != kb[_lang].doc_id.tolist() or dense_index[_lang].ntotal != len(kb[_lang]) or _meta["kb_hash"] != _hash or _meta["model"] != EMBEDDING_MODEL:
        raise ValueError("Indeks demo tidak cocok dengan KB. Jalankan 12_build_demo_indexes.py.")

print(
    "FAISS EN:",
    dense_index["en"].ntotal
)

print(
    "FAISS ID:",
    dense_index["id"].ntotal
)


# ============================================================
# LOAD BGE RERANKER
# ============================================================

print("\nLoading BGE reranker...")

reranker = BGEReranker(
    model_name=RERANKER_MODEL,
    batch_size=RERANKER_BATCH_SIZE,
    max_length=RERANKER_MAX_LENGTH,
)

print("BGE reranker ready.")


# ============================================================
# LOAD SAILOR
# ============================================================

print("\nLoading Sailor2...")

generator = SailorGenerator(
    GENERATOR_MODEL
)

print("Sailor2 ready.")


# ============================================================
# BM25 SEARCH
# ============================================================

def search_bm25(
    question: str,
    language: str,
):

    scores = bm25[
        language
    ].get_scores(
        lexical_tokens(question)
    )

    indices = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), str(kb[language].iloc[i]["doc_id"])))[:BM25_TOP_N]

    results = []

    for rank, idx in enumerate(
        indices,
        start=1,
    ):

        results.append(
            {
                "doc_id":
                    kb[language]
                    .iloc[idx]["doc_id"],

                "rank":
                    rank,

                "score":
                    float(scores[idx]),
            }
        )

    return results


# ============================================================
# DENSE SEARCH
# ============================================================

def search_dense(
    question: str,
    language: str,
):

    query_embedding = encoder.encode(
        [
            "query: "
            + question
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, indices = (
        dense_index[
            language
        ].search(
            query_embedding,
            DENSE_TOP_N,
        )
    )

    results = []

    for rank, (
        idx,
        score,
    ) in enumerate(
        zip(
            indices[0],
            scores[0],
        ),
        start=1,
    ):

        if idx < 0:
            continue

        results.append(
            {
                "doc_id":
                    kb[language]
                    .iloc[int(idx)]["doc_id"],

                "rank":
                    rank,

                "score":
                    float(score),
            }
        )

    return results


# ============================================================
# RRF
# ============================================================

def reciprocal_rank_fusion(
    bm25_results,
    dense_results,
):

    scores = {}

    for results in [
        bm25_results,
        dense_results,
    ]:

        for item in results:

            doc_id = item["doc_id"]

            scores[doc_id] = (
                scores.get(
                    doc_id,
                    0.0
                )
                +
                1.0
                /
                (
                    RRF_K
                    +
                    item["rank"]
                )
            )

    ranked = sorted(
        scores.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    )

    return [
        {
            "doc_id": doc_id,
            "rank": rank,
            "score": float(score),
        }
        for rank, (
            doc_id,
            score,
        ) in enumerate(
            ranked,
            start=1,
        )
    ]


# ============================================================
# HYBRID + RRF + BGE RERANKER
# ============================================================

def search_hybrid_rerank(
    question: str,
    language: str,
    bm25_results,
    dense_results,
):

    # --------------------------------------------------------
    # 1. RRF
    # --------------------------------------------------------

    rrf_results = (
        reciprocal_rank_fusion(
            bm25_results,
            dense_results,
        )
    )

    # --------------------------------------------------------
    # 2. Ambil Top-10 kandidat
    # --------------------------------------------------------

    candidates = (
        rrf_results[
            :RRF_CANDIDATE_TOP_N
        ]
    )

    # --------------------------------------------------------
    # 3. BGE reranking
    # --------------------------------------------------------

    reranked = reranker.rerank(
        query=question,
        candidates=candidates,
        doc_text=doc_text[language],
    )

    # --------------------------------------------------------
    # 4. Format hasil final
    # --------------------------------------------------------

    final_results = []

    for item in reranked:

        final_results.append(
            {
                "doc_id":
                    item["doc_id"],

                "rank":
                    item["rank"],

                "score":
                    item["reranker_score"],

                "rrf_rank":
                    item["rrf_rank"],

                "rrf_score":
                    item["rrf_score"],

                "reranker_score":
                    item["reranker_score"],
            }
        )

    return final_results


# ============================================================
# GET EVIDENCE
# ============================================================

def get_evidence(
    results,
    language,
):

    kb_index = (
        kb[language]
        .set_index("doc_id")
    )

    top_results = (
        results[
            :TOP_K_EVIDENCE
        ]
    )

    evidence = []
    evidence_info = []

    for item in top_results:

        doc_id = item["doc_id"]

        row = kb_index.loc[
            doc_id
        ]

        evidence.append(
            str(
                row["content"]
            )
        )

        evidence_info.append(
            {
                "doc_id":
                    doc_id,

                "content_id":
                    row["content_id"],

                "rank":
                    item["rank"],

                "score":
                    item["score"],

                "rrf_rank":
                    item.get(
                        "rrf_rank"
                    ),

                "rrf_score":
                    item.get(
                        "rrf_score"
                    ),

                "reranker_score":
                    item.get(
                        "reranker_score"
                    ),

                "content":
                    row["content"],
            }
        )

    return (
        evidence,
        evidence_info,
    )


# ============================================================
# FORMAT EVIDENCE
# ============================================================

def format_evidence(
    evidence_info,
    method_label,
):

    evidence_md = ""

    for i, item in enumerate(
        evidence_info,
        start=1,
    ):

        evidence_md += (
            f"### Evidence {i}\n"
            f"**Document:** `{item['doc_id']}`  \n"
            f"**Content ID:** `{item['content_id']}`  \n"
        )

        # S3 mempunyai informasi tambahan
        if (
            method_label
            == "Hybrid RRF + BGE Reranker"
        ):

            evidence_md += (
                f"**RRF Rank:** "
                f"`{int(item['rrf_rank'])}`  \n"

                f"**RRF Score:** "
                f"`{item['rrf_score']:.6f}`  \n"

                f"**Reranker Score:** "
                f"`{item['reranker_score']:.6f}`  \n"
            )

        else:

            evidence_md += (
                f"**Retrieval Score:** "
                f"`{item['score']:.6f}`  \n"
            )

        evidence_md += (
            f"\n{item['content']}\n\n"
            "---\n\n"
        )

    return evidence_md


# ============================================================
# MAIN QA FUNCTION
# ============================================================

def ask_question(
    question,
    method_label,
):

    question = str(
        question
    ).strip()

    if not question:

        return (
            "Silakan masukkan pertanyaan.",
            "",
            "",
        )

    language, _ = detect_language(question)

    # ========================================================
    # RETRIEVAL
    # ========================================================

    bm25_results = search_bm25(
        question,
        language,
    )

    dense_results = search_dense(
        question,
        language,
    )

    if method_label == "BM25":

        results = bm25_results

        process_text = (
            "BM25 → Top-3 Evidence → Sailor2"
        )

    elif method_label == "Dense E5":

        results = dense_results

        process_text = (
            "Multilingual-E5 + FAISS "
            "→ Top-3 Evidence → Sailor2"
        )

    else:

        results = search_hybrid_rerank(
            question=question,
            language=language,
            bm25_results=bm25_results,
            dense_results=dense_results,
        )

        process_text = (
            "BM25 Top-50 + Dense Top-50 "
            "→ RRF "
            "→ Top-10 Candidate "
            "→ BGE Reranker "
            "→ Top-3 Evidence "
            "→ Sailor2"
        )

    # ========================================================
    # TOP-3 EVIDENCE
    # ========================================================

    (
        evidence,
        evidence_info,
    ) = get_evidence(
        results,
        language,
    )

    # ========================================================
    # GENERATION
    # ========================================================

    answer = generator.answer(
        question=question,
        evidence=evidence,
        language=language,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # ========================================================
    # FORMAT EVIDENCE
    # ========================================================

    evidence_md = format_evidence(
        evidence_info,
        method_label,
    )

    return (
        answer,
        f"Bahasa terdeteksi: {language.upper()}\n\n{process_text}",
        evidence_md,
    )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="Bilingual Health Generative QA"
) as demo:

    gr.Markdown(
        """
# Bilingual Health Generative Question Answering

Sistem tanya-jawab kesehatan bilingual
Indonesia–Inggris berbasis
**Retrieval-Augmented Generation (RAG)**.

Metode utama penelitian:

**BM25 + Multilingual-E5 + RRF + BGE Evidence Reranking**

Model generatif:

**Sailor2-1B-Chat**
"""
    )

    with gr.Row():

        gr.Markdown("Bahasa Indonesia atau Inggris dikenali otomatis dari pertanyaan.")

        method = gr.Dropdown(
            choices=[
                "Hybrid RRF + BGE Reranker",
                "Dense E5",
                "BM25",
            ],
            value=(
                "Hybrid RRF + BGE Reranker"
            ),
            label="Metode Retrieval",
        )

    question = gr.Textbox(
        label="Pertanyaan",
        placeholder=(
            "Contoh: Apa itu rheumatoid arthritis?"
        ),
        lines=3,
    )

    ask_button = gr.Button(
        "Tanyakan"
    )

    gr.Markdown(
        "## Jawaban"
    )

    answer = gr.Markdown()

    gr.Markdown(
        "## Proses"
    )

    process = gr.Markdown()

    gr.Markdown(
        "## Retrieved Evidence"
    )

    evidence = gr.Markdown()

    ask_button.click(
        fn=ask_question,
        inputs=[
            question,
            method,
        ],
        outputs=[
            answer,
            process,
            evidence,
        ],
    )

    question.submit(
        fn=ask_question,
        inputs=[
            question,
            method,
        ],
        outputs=[
            answer,
            process,
            evidence,
        ],
    )


if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )