import _bootstrap  # noqa: F401

import json
import torch
import faiss
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.common import load_config, resolve_path


cfg = load_config()

KB_FILES = {
    "en": resolve_path(cfg["paths"]["kb_en_csv"]),
    "id": resolve_path(cfg["paths"]["kb_id_csv"]),
}

MODEL_NAME = cfg["models"]["embedding"]
BATCH_SIZE = cfg["experiment"]["embedding_batch_size"]

INDEX_DIR = resolve_path(cfg["paths"]["index_dir"])
INDEX_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("BUILD PERSISTENT DENSE INDEX")
print("=" * 70)
print("Model :", MODEL_NAME)
print("Device:", device)

encoder = SentenceTransformer(
    MODEL_NAME,
    device=device
)

encoder.max_seq_length = 512

for lang, kb_path in KB_FILES.items():

    print("\n" + "=" * 70)
    print("LANGUAGE:", lang.upper())
    print("=" * 70)

    kb = pd.read_csv(kb_path)

    passages = [
        "passage: " + str(x)
        for x in kb["content"].tolist()
    ]

    print("Total documents:", len(passages))

    embeddings = encoder.encode(
        passages,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    index_path = (
        INDEX_DIR
        / f"multilingual_e5_{lang}.faiss"
    )

    faiss.write_index(
        index,
        str(index_path)
    )

    # Simpan mapping urutan FAISS → doc_id
    mapping = kb[
        [
            "doc_id",
            "content_id",
            "parent_id",
        ]
    ].copy()

    mapping_path = (
        INDEX_DIR
        / f"multilingual_e5_{lang}_mapping.csv"
    )

    mapping.to_csv(
        mapping_path,
        index=False
    )

    import hashlib
    meta = {"model": MODEL_NAME, "max_seq_length": 512,
            "kb_hash": hashlib.sha256(pd.util.hash_pandas_object(kb, index=False).values.tobytes()).hexdigest()}
    (INDEX_DIR / f"multilingual_e5_{lang}_meta.json").write_text(json.dumps(meta, indent=2))
    print("Index :", index_path)
    print("Map   :", mapping_path)
    print("Vectors:", index.ntotal)
    print("Dimension:", index.d)

print("\nSUKSES.")
