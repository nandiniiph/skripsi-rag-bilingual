from __future__ import annotations

from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from .common import ensure_parent


def clean_whitespace(text: str) -> str:
    """
    Merapikan whitespace tanpa mengubah isi teks.
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Memecah teks menjadi unit kalimat.

    Pemisahan utama dilakukan setelah:
    . ? !

    Fungsi ini tidak membutuhkan library tambahan seperti NLTK.
    """
    text = clean_whitespace(text)

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def token_count(
    text: str,
    tokenizer,
) -> int:
    """
    Menghitung jumlah token tanpa special token.
    """
    if not text:
        return 0

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def split_long_text_by_words(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Fallback jika satu kalimat saja melebihi chunk_size.

    Pemotongan dilakukan berdasarkan kata, bukan langsung
    berdasarkan subword token, sehingga chunk tidak dimulai
    di tengah kata.
    """

    words = text.split()

    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):

        current_words: list[str] = []
        end = start

        while end < len(words):
            candidate_words = current_words + [words[end]]
            candidate = " ".join(candidate_words)

            count = token_count(
                candidate,
                tokenizer,
            )

            if count > chunk_size:
                break

            current_words.append(words[end])
            end += 1

        # Safety jika satu kata sangat panjang
        if not current_words:
            current_words = [words[start]]
            end = start + 1

        chunk = " ".join(current_words).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        # Cari overlap berdasarkan kata dengan target
        # maksimal kurang lebih overlap token.
        overlap_words: list[str] = []

        i = end - 1

        while i >= start:
            candidate = [words[i]] + overlap_words
            candidate_text = " ".join(candidate)

            if token_count(
                candidate_text,
                tokenizer,
            ) > overlap:
                break

            overlap_words = candidate
            i -= 1

        # Pastikan selalu maju agar tidak infinite loop
        new_start = end - len(overlap_words)

        if new_start <= start:
            new_start = end

        start = new_start

    return chunks


def create_sentence_chunks(
    text: str,
    tokenizer,
    chunk_size: int = 256,
    overlap: int = 64,
) -> list[str]:
    """
    Membentuk chunk dengan prioritas batas kalimat.

    Strategi:
    1. Teks dibagi menjadi kalimat.
    2. Kalimat digabung sampai mendekati chunk_size.
    3. Ketika chunk penuh, sebagian kalimat terakhir
       digunakan sebagai overlap.
    4. Jika satu kalimat > chunk_size, gunakan fallback
       pemotongan berbasis kata.
    """

    sentences = split_sentences(text)

    if not sentences:
        return []

    # Pecah kalimat yang sendirian terlalu panjang
    units: list[str] = []

    for sentence in sentences:

        if token_count(
            sentence,
            tokenizer,
        ) <= chunk_size:

            units.append(sentence)

        else:

            long_parts = split_long_text_by_words(
                text=sentence,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            units.extend(long_parts)

    chunks: list[str] = []
    current_units: list[str] = []

    i = 0

    while i < len(units):

        unit = units[i]

        candidate_units = current_units + [unit]
        candidate_text = " ".join(candidate_units)

        candidate_tokens = token_count(
            candidate_text,
            tokenizer,
        )

        if candidate_tokens <= chunk_size:
            current_units.append(unit)
            i += 1
            continue

        # Simpan current chunk
        if current_units:

            chunk_text = " ".join(current_units).strip()

            if chunk_text:
                chunks.append(chunk_text)

            # Membentuk overlap dari akhir chunk sebelumnya
            overlap_units: list[str] = []

            for previous_unit in reversed(current_units):

                candidate_overlap = (
                    [previous_unit] + overlap_units
                )

                overlap_text = " ".join(
                    candidate_overlap
                )

                if token_count(
                    overlap_text,
                    tokenizer,
                ) > overlap:
                    break

                overlap_units = candidate_overlap

            current_units = overlap_units

            # Jika overlap membuat unit baru tetap tidak muat,
            # hapus overlap agar proses tetap berjalan
            test_text = " ".join(
                current_units + [unit]
            )

            if (
                current_units
                and token_count(
                    test_text,
                    tokenizer,
                ) > chunk_size
            ):
                current_units = []

        else:
            # Seharusnya hanya terjadi pada edge case.
            # Unit sangat panjang dipecah sekali lagi.
            parts = split_long_text_by_words(
                text=unit,
                tokenizer=tokenizer,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            chunks.extend(parts)
            i += 1

    # Simpan chunk terakhir
    if current_units:
        final_chunk = " ".join(
            current_units
        ).strip()

        if final_chunk:
            chunks.append(final_chunk)

    # Hilangkan duplicate chunk yang mungkin terjadi
    result: list[str] = []

    for chunk in chunks:
        chunk = clean_whitespace(chunk)

        if not chunk:
            continue

        if result and chunk == result[-1]:
            continue

        result.append(chunk)

    return result


def build_chunks(
    qa_df: pd.DataFrame,
    model_name: str,
    chunk_size: int = 256,
    overlap: int = 64,
) -> pd.DataFrame:

    if overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap harus lebih kecil "
            "daripada chunk_size"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    rows = []

    for row in tqdm(
        qa_df.itertuples(index=False),
        total=len(qa_df),
        desc="Sentence-aware chunking",
    ):

        answer = clean_whitespace(
            str(row.answer)
        )

        chunks = create_sentence_chunks(
            text=answer,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        token_cursor = 0

        for n, piece in enumerate(
            chunks,
            start=1,
        ):

            count = token_count(
                piece,
                tokenizer,
            )

            content_id = (
                f"{row.parent_id}-C{n:02d}"
            )

            rows.append(
                {
                    "parent_id": row.parent_id,
                    "content_id": content_id,
                    "content": piece,
                    "source": row.source,
                    "focus_area": row.focus_area,

                    # token_start/token_end sekarang
                    # hanya metadata estimasi posisi.
                    "token_start": token_cursor,
                    "token_end": token_cursor + count,
                    "token_count": count,
                }
            )

            # Karena ada overlap, cursor hanya
            # maju sebesar token baru.
            token_cursor += max(
                count - overlap,
                1,
            )

    result = pd.DataFrame(rows)

    # Validasi dasar
    if not result.empty:

        if result["content"].isna().any():
            raise ValueError(
                "Ditemukan chunk kosong."
            )

        if (
            result["token_count"] > chunk_size
        ).any():

            bad = result[
                result["token_count"]
                > chunk_size
            ]

            raise ValueError(
                f"Ditemukan {len(bad)} chunk "
                f"melebihi {chunk_size} token."
            )

    return result


def save_chunks(
    df: pd.DataFrame,
    path: Path,
) -> None:

    ensure_parent(path)

    df.to_csv(
        path,
        index=False,
    )