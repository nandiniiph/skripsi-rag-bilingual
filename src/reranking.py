from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BGEReranker:
    """
    Multilingual cross-encoder reranker.

    Model menerima pasangan:
        question + passage

    kemudian menghasilkan relevance score.

    Score digunakan hanya untuk ranking, bukan sebagai probabilitas.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading reranker: {model_name}")
        print(f"Reranker device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, query: str, passages: list[str]) -> list[float]:
        """
        Memberikan relevance score untuk setiap passage terhadap query.
        """

        if not passages:
            return []

        all_scores = []
        self.last_input_lengths = []

        for start in range(0, len(passages), self.batch_size):
            batch_passages = passages[
                start : start + self.batch_size
            ]

            batch_queries = [query] * len(batch_passages)

            lengths = [len(self.tokenizer(q, p, truncation=False)["input_ids"]) for q, p in zip(batch_queries, batch_passages)]
            self.last_input_lengths.extend(lengths)
            inputs = self.tokenizer(
                batch_queries,
                batch_passages,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            outputs = self.model(
                **inputs,
                return_dict=True,
            )

            scores = (
                outputs.logits
                .view(-1)
                .float()
                .cpu()
                .tolist()
            )

            all_scores.extend(scores)

        return all_scores

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        doc_text: dict[str, str],
    ) -> list[dict]:
        """
        candidates berasal dari ranking RRF.

        Kandidat akan dinilai ulang menggunakan BGE reranker.
        """

        passages = []

        for candidate in candidates:
            doc_id = candidate["doc_id"]

            if doc_id not in doc_text:
                raise KeyError(
                    f"Document {doc_id} tidak ditemukan "
                    "di knowledge base."
                )

            passages.append(
                str(doc_text[doc_id])
            )

        reranker_scores = self.score(
            query,
            passages,
        )

        reranked = []

        for i, (candidate, reranker_score) in enumerate(zip(candidates, reranker_scores)):
            reranked.append(
                {
                    "doc_id": candidate["doc_id"],
                    "reranker_input_tokens": self.last_input_lengths[i],
                    "reranker_truncated": self.last_input_lengths[i] > self.max_length,

                    # Ranking sebelum BGE
                    "rrf_rank": candidate["rank"],
                    "rrf_score": candidate["score"],

                    # Score BGE
                    "reranker_score": float(
                        reranker_score
                    ),
                }
            )

        reranked.sort(
            key=lambda x: (
                -x["reranker_score"],
                x["doc_id"],
            )
        )

        for rank, item in enumerate(
            reranked,
            start=1,
        ):
            item["rank"] = rank

        return reranked