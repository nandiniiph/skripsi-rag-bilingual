import _bootstrap  # noqa: F401

import pandas as pd

from src.common import (
    load_config,
    resolve_path,
    setup_logging,
    seed_everything,
)

from src.retrieval import (
    run_all_retrieval,
)


cfg = load_config()
import subprocess, sys
# A new retrieval run must not silently replace the ranking used by generation checkpoints.
if (resolve_path(cfg['paths']['retrieval_dir']) / 'retrieval_results.csv').exists():
    raise FileExistsError('Hasil retrieval sudah ada. Gunakan hasil ini atau ubah seluruh jalur output untuk eksperimen baru.')
subprocess.run([sys.executable, str(__import__('pathlib').Path(__file__).with_name('08_check_inputs.py'))], check=True)

seed_everything(
    cfg["project"]["seed"]
)

log = setup_logging(
    resolve_path(
        cfg["paths"]["log_file"]
    )
)

e = cfg["experiment"]


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


out = run_all_retrieval(
    qa_eval=qa,

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

    output_dir=
        resolve_path(
            cfg["paths"]["retrieval_dir"]
        ),

    embedding_batch_size=
        e["embedding_batch_size"],

    reranker_batch_size=
        e["reranker_batch_size"],

    reranker_max_length=
        e["reranker_max_length"],
)


log.info(
    "Retrieval + reranking selesai: %d rows",
    len(out),
)