import _bootstrap  # noqa: F401
import pandas as pd
from src.common import load_config, resolve_path, setup_logging
from src.chunking import build_chunks, save_chunks
cfg=load_config(); log=setup_logging(resolve_path(cfg["paths"]["log_file"]))
df=pd.read_csv(resolve_path(cfg["paths"]["cleaned_qa_en_csv"]))
p=cfg["preprocessing"]
out=build_chunks(df,cfg["models"]["chunk_tokenizer"],p["chunk_size"],p["chunk_overlap"])
save_chunks(out,resolve_path(cfg["paths"]["chunks_en_csv"]))
log.info("Chunking selesai: %d chunks",len(out))
