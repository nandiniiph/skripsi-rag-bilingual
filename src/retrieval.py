from __future__ import annotations

import re
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .common import ensure_parent
from .reranking import BGEReranker
from .language import detect_questions
from .validation import validate_inputs


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def lexical_tokens(text: str):
    return TOKEN_RE.findall(str(text).casefold())


# =========================================================
# S1 - BM25
# =========================================================

class BM25Retriever:
    def __init__(self, kb: pd.DataFrame):
        self.kb = kb.reset_index(drop=True)

        self.tokens = [
            lexical_tokens(x)
            for x in self.kb["content"]
        ]

        self.model = BM25Okapi(self.tokens, k1=1.5, b=0.75, epsilon=0.25)

    def search(self, query: str, top_n: int = 50):
        scores = self.model.get_scores(
            lexical_tokens(query)
        )

        idx = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), str(self.kb.iloc[i]["doc_id"])))[:top_n]

        return [
            {
                "doc_id": self.kb.iloc[i]["doc_id"],
                "rank": rank,
                "score": float(scores[i]),
            }
            for rank, i in enumerate(idx, start=1)
        ]


# =========================================================
# S2 - DENSE RETRIEVAL
# =========================================================

class DenseRetriever:
    def __init__(
        self,
        kb: pd.DataFrame,
        encoder: SentenceTransformer,
        batch_size: int = 32,
    ):
        self.kb = kb.reset_index(drop=True)
        self.encoder = encoder
        self.encoder.max_seq_length = 512

        passages = [
            "passage: " + str(x)
            for x in self.kb["content"]
        ]

        self.input_token_lengths = [len(self.encoder.tokenizer(text, truncation=False)["input_ids"]) for text in passages]
        embeddings = self.encoder.encode(
            passages,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        self.index.add(embeddings)

    def search(self, query: str, top_n: int = 50):
        query_embedding = self.encoder.encode(
            ["query: " + str(query)],
            normalize_embeddings=True,
        ).astype("float32")

        scores, idx = self.index.search(query_embedding, min(top_n, len(self.kb)))

        results = []

        for rank, (i, score) in enumerate(
            zip(idx[0], scores[0]),
            start=1,
        ):
            if i < 0:
                continue

            results.append(
                {
                    "doc_id": self.kb.iloc[int(i)]["doc_id"],
                    "rank": rank,
                    "score": float(score),
                }
            )

        return results


# =========================================================
# RRF
# =========================================================

def rrf(
    bm25_results,
    dense_results,
    k: int = 60,
):
    scores = {}

    for results in (
        bm25_results,
        dense_results,
    ):
        for item in results:
            doc_id = item["doc_id"]

            scores[doc_id] = (
                scores.get(doc_id, 0.0)
                + 1.0 / (k + item["rank"])
            )

    ranked = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    return [
        {
            "doc_id": doc_id,
            "rank": rank,
            "score": float(score),
        }
        for rank, (doc_id, score)
        in enumerate(ranked, start=1)
    ]


# =========================================================
# SEMUA SKENARIO
# =========================================================

def run_all_retrieval(
    qa_eval: pd.DataFrame,
    kb_en: pd.DataFrame,
    kb_id: pd.DataFrame,

    embedding_model: str,
    reranker_model: str,

    bm25_top_n: int,
    dense_top_n: int,
    rrf_k: int,
    rrf_candidate_top_n: int,

    output_dir: Path,

    embedding_batch_size: int = 32,
    reranker_batch_size: int = 8,
    reranker_max_length: int = 512,
):
    validate_inputs(qa_eval, kb_en, kb_id)
    ensure_parent(output_dir / "dummy")
    detections = detect_questions(qa_eval)
    detections.to_csv(output_dir / "language_detection.csv", index=False)
    predictions = detections.set_index("qa_id")["predicted_language"].to_dict()

    # =====================================================
    # Load E5 SATU KALI
    # =====================================================

    print("\nLoading embedding model...")
    encoder = SentenceTransformer(
        embedding_model
    )

    # =====================================================
    # Retriever Indonesia & Inggris
    # =====================================================

    retrievers = {}

    for lang, kb in [
        ("en", kb_en),
        ("id", kb_id),
    ]:
        print(
            f"\nBuilding retrievers for language: {lang}"
        )

        retrievers[
            (lang, "bm25")
        ] = BM25Retriever(kb)

        retrievers[
            (lang, "dense")
        ] = DenseRetriever(
            kb=kb,
            encoder=encoder,
            batch_size=embedding_batch_size,
        )

    # Record actual model revisions and every E5 truncation. No silent correction of the corpus.
    audit = []
    for lang, kb in [("en", kb_en), ("id", kb_id)]:
        dense = retrievers[(lang, "dense")]
        for doc_id, n in zip(kb.doc_id, getattr(dense, "input_token_lengths", [])):
            audit.append({"kind":"passage", "id":doc_id, "language":lang, "tokens":n, "truncated":n>512})
    if hasattr(encoder, "tokenizer"):
        for q in qa_eval.itertuples(index=False):
            n = len(encoder.tokenizer("query: " + str(q.question), truncation=False)["input_ids"])
            audit.append({"kind":"query", "id":q.qa_id, "language":q.language, "tokens":n, "truncated":n>512})
    pd.DataFrame(audit, columns=["kind","id","language","tokens","truncated"]).to_csv(output_dir / "e5_token_audit.csv", index=False)

    # =====================================================
    # Lookup content berdasarkan doc_id
    # =====================================================

    doc_text = {
        "en": (
            kb_en
            .set_index("doc_id")["content"]
            .astype(str)
            .to_dict()
        ),
        "id": (
            kb_id
            .set_index("doc_id")["content"]
            .astype(str)
            .to_dict()
        ),
    }

    # =====================================================
    # Load BGE reranker
    # =====================================================

    reranker = BGEReranker(
        model_name=reranker_model,
        batch_size=reranker_batch_size,
        max_length=reranker_max_length,
    )

    revisions = {"embedding_model":embedding_model, "reranker_model":reranker_model}
    if hasattr(encoder, "_first_module"):
        revisions["embedding_revision"] = getattr(encoder._first_module().auto_model.config, "_commit_hash", None)
    revisions["reranker_revision"] = getattr(getattr(getattr(reranker,"model",None),"config",None),"_commit_hash",None)
    (output_dir / "model_revisions.json").write_text(json.dumps(revisions,indent=2))
    all_rows = []

    # Hasil RRF sebelum BGE,
    # hanya untuk analisis internal S3
    rrf_before_rows = []

    # =====================================================
    # Retrieval setiap query
    # =====================================================

    for q in tqdm(
        qa_eval.itertuples(index=False),
        total=len(qa_eval),
        desc="Retrieval + Reranking",
    ):

        true_language = q.language
        language = predictions[q.qa_id]
        question = str(q.question)

        # =================================================
        # S1 - BM25
        # =================================================

        bm25_results = retrievers[
            (language, "bm25")
        ].search(
            question,
            top_n=bm25_top_n,
        )

        # =================================================
        # S2 - Dense
        # =================================================

        dense_results = retrievers[
            (language, "dense")
        ].search(
            question,
            top_n=dense_top_n,
        )

        # =================================================
        # S3 - Hybrid RRF
        # =================================================

        rrf_results = rrf(
            bm25_results,
            dense_results,
            k=rrf_k,
        )

        # Simpan ranking RRF sebelum reranking
        for item in rrf_results[:50]:
            rrf_before_rows.append(
                {
                    "qa_id": q.qa_id,
                    "pair_id": q.pair_id,
                    "language": true_language,
                    "predicted_language": language,
                    "doc_id": item["doc_id"],
                    "rank": item["rank"],
                    "rrf_score": item["score"],
                }
            )

        # =================================================
        # Top-10 RRF → BGE reranker
        # =================================================

        candidates = rrf_results[
            :rrf_candidate_top_n
        ]

        reranked_candidates = reranker.rerank(
            query=question,
            candidates=candidates,
            doc_text=doc_text[language],
        )

        candidate_ids = {
            item["doc_id"]
            for item in candidates
        }

        # Dokumen setelah posisi Top-10 RRF
        # tetap dipertahankan urutannya.
        remaining_results = [
            item
            for item in rrf_results
            if item["doc_id"] not in candidate_ids
        ]

        # =================================================
        # Ranking final S3
        # =================================================

        hybrid_final = []

        # Top-N yang sudah direrank BGE
        for item in reranked_candidates:
            hybrid_final.append(
                {
                    "doc_id": item["doc_id"],
                    "rrf_rank": item["rrf_rank"],
                    "rrf_score": item["rrf_score"],
                    "reranker_score":
                        item["reranker_score"],
                    "reranker_input_tokens": item.get("reranker_input_tokens"),
                    "reranker_truncated": item.get("reranker_truncated"),
                }
            )

        # Sisanya mengikuti ranking RRF
        for item in remaining_results:
            hybrid_final.append(
                {
                    "doc_id": item["doc_id"],
                    "rrf_rank": item["rank"],
                    "rrf_score": item["score"],
                    "reranker_score": None,
                }
            )

        # Beri ranking final
        for new_rank, item in enumerate(
            hybrid_final,
            start=1,
        ):
            item["rank"] = new_rank

        # =================================================
        # SAVE S1
        # =================================================

        for item in bm25_results:
            all_rows.append(
                {
                    "qa_id": q.qa_id,
                    "pair_id": q.pair_id,
                    "language": true_language,
                    "predicted_language": language,

                    "scenario": "S1_BM25",

                    "doc_id": item["doc_id"],
                    "rank": item["rank"],
                    "score": item["score"],

                    "rrf_rank": None,
                    "rrf_score": None,
                    "reranker_score": None,
                }
            )

        # =================================================
        # SAVE S2
        # =================================================

        for item in dense_results:
            all_rows.append(
                {
                    "qa_id": q.qa_id,
                    "pair_id": q.pair_id,
                    "language": true_language,
                    "predicted_language": language,

                    "scenario": "S2_DENSE",

                    "doc_id": item["doc_id"],
                    "rank": item["rank"],
                    "score": item["score"],

                    "rrf_rank": None,
                    "rrf_score": None,
                    "reranker_score": None,
                }
            )

        # =================================================
        # SAVE S3
        # =================================================

        for item in hybrid_final[:50]:

            # Kolom score dipertahankan agar kompatibel
            # dengan struktur pipeline lama.
            if item["reranker_score"] is not None:
                final_score = item["reranker_score"]
            else:
                final_score = item["rrf_score"]

            all_rows.append(
                {
                    "qa_id": q.qa_id,
                    "pair_id": q.pair_id,
                    "language": true_language,
                    "predicted_language": language,

                    "scenario":
                        "S3_HYBRID_RRF_RERANK",

                    "doc_id": item["doc_id"],
                    "rank": item["rank"],
                    "score": final_score,

                    "rrf_rank": item["rrf_rank"],
                    "rrf_score": item["rrf_score"],
                    "reranker_score":
                        item["reranker_score"],
                    "reranker_input_tokens": item.get("reranker_input_tokens"),
                    "reranker_truncated": item.get("reranker_truncated"),
                }
            )

    # =====================================================
    # SIMPAN OUTPUT UTAMA
    # =====================================================

    output = pd.DataFrame(all_rows)

    retrieval_path = (
        output_dir / "retrieval_results.csv"
    )

    ensure_parent(retrieval_path)

    output.to_csv(
        retrieval_path,
        index=False,
    )

    # =====================================================
    # SIMPAN RRF SEBELUM RERANKING
    # =====================================================

    rrf_before = pd.DataFrame(
        rrf_before_rows
    )

    rrf_before_path = (
        output_dir /
        "s3_rrf_before_rerank.csv"
    )

    rrf_before.to_csv(
        rrf_before_path,
        index=False,
    )

    print(
        f"\nRetrieval results: {retrieval_path}"
    )

    print(
        f"RRF before reranking: {rrf_before_path}"
    )

    return output