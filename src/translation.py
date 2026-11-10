from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from .common import ensure_parent


# ============================================================
# DEVICE
# ============================================================

def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_space(text: str) -> str:
    """
    Merapikan whitespace tanpa mengubah makna teks.
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Memecah teks menjadi unit kalimat untuk keperluan
    translasi internal.

    content_id tidak berubah.
    """

    text = normalize_space(text)

    if not text:
        return []

    parts = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        p.strip()
        for p in parts
        if p.strip()
    ]


def token_count(
    text: str,
    tokenizer,
) -> int:

    if not text:
        return 0

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def split_long_sentence_by_words(
    sentence: str,
    tokenizer,
    max_tokens: int,
) -> list[str]:
    """
    Fallback apabila satu kalimat lebih panjang dari
    batas input NLLB.

    Pemotongan dilakukan pada batas kata.
    """

    words = sentence.split()

    if not words:
        return []

    pieces: list[str] = []
    current: list[str] = []

    for word in words:

        candidate = " ".join(
            current + [word]
        )

        if (
            current
            and token_count(
                candidate,
                tokenizer,
            ) > max_tokens
        ):
            pieces.append(
                " ".join(current).strip()
            )

            current = [word]

        else:
            current.append(word)

    if current:
        pieces.append(
            " ".join(current).strip()
        )

    return pieces


def segment_for_translation(
    text: str,
    tokenizer,
    max_segment_tokens: int = 220,
) -> list[str]:
    """
    Memecah satu chunk menjadi beberapa bagian pendek
    untuk diterjemahkan oleh NLLB.

    Strategi:
    - prioritas batas kalimat;
    - satu segment maksimum sekitar 220 token NLLB;
    - hasil translation seluruh segment digabung kembali.

    Ini TIDAK mengubah chunk/content_id pada knowledge base.
    """

    text = normalize_space(text)

    if not text:
        return [""]

    sentences = split_sentences(text)

    units: list[str] = []

    for sentence in sentences:

        if (
            token_count(
                sentence,
                tokenizer,
            )
            <= max_segment_tokens
        ):
            units.append(sentence)

        else:
            units.extend(
                split_long_sentence_by_words(
                    sentence,
                    tokenizer,
                    max_segment_tokens,
                )
            )

    segments: list[str] = []
    current: list[str] = []

    for unit in units:

        candidate = " ".join(
            current + [unit]
        ).strip()

        if (
            current
            and token_count(
                candidate,
                tokenizer,
            )
            > max_segment_tokens
        ):

            segments.append(
                " ".join(current).strip()
            )

            current = [unit]

        else:
            current.append(unit)

    if current:
        segments.append(
            " ".join(current).strip()
        )

    return segments


# ============================================================
# NLLB TRANSLATOR
# ============================================================

class NLLBTranslator:

    def __init__(
        self,
        model_name: str,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "ind_Latn",
        device: str | None = None,
    ):

        self.device = (
            device or choose_device()
        )

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
                src_lang=src_lang,
            )
        )

        self.tokenizer.src_lang = src_lang

        dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.model = (
            AutoModelForSeq2SeqLM
            .from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.tgt_id = (
            self.tokenizer
            .convert_tokens_to_ids(
                tgt_lang
            )
        )

        if self.tgt_id is None:
            raise ValueError(
                f"Target language token "
                f"{tgt_lang} tidak ditemukan."
            )

        print(
            f"NLLB device : {self.device}"
        )


    @torch.inference_mode()
    def translate_segments(
        self,
        texts: list[str],
        max_input_tokens: int = 256,
    ) -> list[str]:
        """
        Menerjemahkan satu batch segment pendek.

        Berbeda dengan versi lama:
        - beam search
        - repetition penalty
        - no-repeat ngram
        - output length dibatasi
        """

        if not texts:
            return []

        cleaned = [
            normalize_space(text)
            for text in texts
        ]

        enc = self.tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )

        enc = {
            key: value.to(self.device)
            for key, value in enc.items()
        }

        # Menghitung panjang input terpanjang
        source_lengths = (
            enc["attention_mask"]
            .sum(dim=1)
        )

        max_source_len = int(
            source_lengths.max().item()
        )

        # Output translation tidak dibiarkan sampai
        # 512 token seperti kode sebelumnya.
        max_output_tokens = min(
            384,
            max(
                64,
                int(
                    max_source_len * 1.5
                    + 32
                ),
            ),
        )

        generated = self.model.generate(
            **enc,

            forced_bos_token_id=(
                self.tgt_id
            ),

            num_beams=4,

            max_new_tokens=(
                max_output_tokens
            ),

            repetition_penalty=1.15,

            no_repeat_ngram_size=4,

            length_penalty=1.0,

            early_stopping=True,
        )

        decoded = (
            self.tokenizer
            .batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )

        return [
            normalize_space(text)
            for text in decoded
        ]


    def translate_batch(
        self,
        texts: list[str],
        max_input_tokens: int = 512,
    ) -> list[str]:
        """
        Interface tetap sama dengan kode lama.

        Namun sekarang setiap document/chunk terlebih
        dahulu dibagi menjadi segment pendek.

        Setelah semua segment diterjemahkan,
        hasilnya digabung kembali.
        """

        final_results: list[str] = []

        # 220 dipilih agar cukup aman terhadap perbedaan
        # tokenisasi E5 dan NLLB.
        max_segment_tokens = min(
            220,
            max_input_tokens - 32,
        )

        for text in texts:

            text = normalize_space(text)

            if not text:
                final_results.append("")
                continue

            segments = segment_for_translation(
                text=text,
                tokenizer=self.tokenizer,
                max_segment_tokens=(
                    max_segment_tokens
                ),
            )

            translated_segments = (
                self.translate_segments(
                    segments,
                    max_input_tokens=(
                        max_input_tokens
                    ),
                )
            )

            final_text = " ".join(
                translated_segments
            ).strip()

            final_results.append(
                final_text
            )

        return final_results


# ============================================================
# TRANSLATE SERIES
# ============================================================

def translate_series(
    series: pd.Series,
    translator: NLLBTranslator,
    batch_size: int,
    max_input_tokens: int,
) -> list[str]:

    texts = (
        series
        .fillna("")
        .astype(str)
        .tolist()
    )

    result: list[str] = []

    for i in tqdm(
        range(
            0,
            len(texts),
            batch_size,
        ),
        desc="Translation",
    ):

        batch = texts[
            i:i + batch_size
        ]

        translated = (
            translator.translate_batch(
                batch,
                max_input_tokens=(
                    max_input_tokens
                ),
            )
        )

        result.extend(translated)

    return result


# ============================================================
# TRANSLATE QA FILE
# ============================================================

def translate_qa_file(
    input_csv: Path,
    output_csv: Path,
    translator: NLLBTranslator,
    batch_size: int = 4,
    max_input_tokens: int = 512,
) -> pd.DataFrame:

    df = pd.read_csv(input_csv)

    out = df[
        [
            "parent_id",
            "source",
            "focus_area",
        ]
    ].copy()

    print(
        "\nMenerjemahkan QUESTION..."
    )

    out["question"] = (
        translate_series(
            df["question"],
            translator,
            batch_size,
            max_input_tokens,
        )
    )

    print(
        "\nMenerjemahkan "
        "REFERENCE ANSWER..."
    )

    out["reference_answer"] = (
        translate_series(
            df["answer"],
            translator,
            batch_size,
            max_input_tokens,
        )
    )

    ensure_parent(output_csv)

    out.to_csv(
        output_csv,
        index=False,
    )

    return out


# ============================================================
# TRANSLATE CHUNKS
# ============================================================

def translate_chunks_file(
    input_csv: Path,
    output_csv: Path,
    translator: NLLBTranslator,
    batch_size: int = 4,
    max_input_tokens: int = 512,
) -> pd.DataFrame:

    df = pd.read_csv(input_csv)

    out = df[
        [
            "parent_id",
            "content_id",
            "source",
            "focus_area",
        ]
    ].copy()

    print(
        "\nMenerjemahkan KNOWLEDGE BASE CHUNKS..."
    )

    out["content"] = (
        translate_series(
            df["content"],
            translator,
            batch_size,
            max_input_tokens,
        )
    )

    ensure_parent(output_csv)

    out.to_csv(
        output_csv,
        index=False,
    )

    return out


# ============================================================
# TRANSLATION QUALITY CHECK
# ============================================================

def has_consecutive_repetition(
    text: str,
    threshold: int = 4,
) -> bool:
    """
    Contoh yang terdeteksi:

    diet diet diet diet
    jaringan jaringan jaringan jaringan
    """

    words = re.findall(
        r"\b[\w'-]+\b",
        str(text).lower(),
    )

    if not words:
        return False

    count = 1

    for i in range(
        1,
        len(words),
    ):

        if words[i] == words[i - 1]:
            count += 1

            if count >= threshold:
                return True

        else:
            count = 1

    return False


def low_word_diversity(
    text: str,
) -> bool:
    """
    Mendeteksi output panjang tetapi hanya mengulang
    sedikit kata.
    """

    words = re.findall(
        r"\b[\w'-]+\b",
        str(text).lower(),
    )

    if len(words) < 30:
        return False

    unique_ratio = (
        len(set(words))
        / len(words)
    )

    return unique_ratio < 0.25


def repeated_ngram_ratio(
    text: str,
    n: int = 4,
) -> float:

    words = re.findall(
        r"\b[\w'-]+\b",
        str(text).lower(),
    )

    if len(words) < n:
        return 0.0

    ngrams = [
        tuple(
            words[i:i + n]
        )
        for i in range(
            len(words) - n + 1
        )
    ]

    if not ngrams:
        return 0.0

    unique = len(
        set(ngrams)
    )

    return 1.0 - (
        unique / len(ngrams)
    )


def detect_translation_issues(
    english_text: str,
    indonesian_text: str,
) -> list[str]:

    en = normalize_space(
        english_text
    )

    ind = normalize_space(
        indonesian_text
    )

    issues: list[str] = []

    if not ind:
        issues.append("empty")
        return issues

    if en.lower() == ind.lower():
        issues.append(
            "same-as-source"
        )

    en_len = max(
        len(en),
        1,
    )

    ratio = (
        len(ind)
        / en_len
    )

    if ratio < 0.35:
        issues.append(
            "translation-too-short"
        )

    if ratio > 2.50:
        issues.append(
            "translation-too-long"
        )

    if has_consecutive_repetition(
        ind,
        threshold=4,
    ):
        issues.append(
            "consecutive-word-repetition"
        )

    if low_word_diversity(ind):
        issues.append(
            "low-word-diversity"
        )

    ngram_repeat = (
        repeated_ngram_ratio(
            ind,
            n=4,
        )
    )

    if ngram_repeat > 0.45:
        issues.append(
            "high-ngram-repetition"
        )

    en_words = len(
        en.split()
    )

    id_words = len(
        ind.split()
    )

    if (
        en_words >= 30
        and id_words <= 5
    ):
        issues.append(
            "severely-truncated"
        )

    return issues


# ============================================================
# BUILD QUALITY REVIEW FILES
# ============================================================

def build_translation_quality_files(
    en_chunks: pd.DataFrame,
    id_chunks: pd.DataFrame,
    issues_path: Path,
    review_path: Path,
    sample_n: int,
    seed: int,
) -> None:

    merged = (
        en_chunks[
            [
                "content_id",
                "content",
            ]
        ]
        .rename(
            columns={
                "content":
                "english_text"
            }
        )
        .merge(
            id_chunks[
                [
                    "content_id",
                    "content",
                ]
            ]
            .rename(
                columns={
                    "content":
                    "indonesian_text"
                }
            ),
            on="content_id",
            how="inner",
        )
    )

    issue_rows = []

    for row in tqdm(
        merged.itertuples(
            index=False
        ),
        total=len(merged),
        desc="Translation quality check",
    ):

        issues = (
            detect_translation_issues(
                row.english_text,
                row.indonesian_text,
            )
        )

        if issues:
            issue_rows.append(
                {
                    "content_id":
                    row.content_id,

                    "english_text":
                    row.english_text,

                    "indonesian_text":
                    row.indonesian_text,

                    "issue":
                    ";".join(issues),
                }
            )

    issues_df = pd.DataFrame(
        issue_rows,
        columns=[
            "content_id",
            "english_text",
            "indonesian_text",
            "issue",
        ],
    )

    ensure_parent(
        issues_path
    )

    issues_df.to_csv(
        issues_path,
        index=False,
    )

    # --------------------------------------------------------
    # RANDOM MANUAL REVIEW SAMPLE
    # --------------------------------------------------------

    n = min(
        sample_n,
        len(merged),
    )

    review = merged.sample(
        n=n,
        random_state=seed,
    ).copy()

    review["is_valid"] = ""
    review["revision"] = ""
    review["notes"] = ""

    ensure_parent(
        review_path
    )

    review.to_csv(
        review_path,
        index=False,
    )

    print(
        "\nTranslation quality summary"
    )

    print(
        "Total translated chunks :",
        len(merged),
    )

    print(
        "Automatic issues        :",
        len(issues_df),
    )

    print(
        "Manual review sample    :",
        len(review),
    )