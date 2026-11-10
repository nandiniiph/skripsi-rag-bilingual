import _bootstrap  # noqa: F401
import pandas as pd
from src.common import load_config, resolve_path, setup_logging
from src.knowledge_base import build_knowledge_bases, save_kb
cfg=load_config(); log=setup_logging(resolve_path(cfg["paths"]["log_file"]))
en=pd.read_csv(resolve_path(cfg["paths"]["chunks_en_csv"])); idf=pd.read_csv(resolve_path(cfg["paths"]["chunks_id_csv"]))
kb_en,kb_id=build_knowledge_bases(en,idf); save_kb(kb_en,resolve_path(cfg["paths"]["kb_en_csv"])); save_kb(kb_id,resolve_path(cfg["paths"]["kb_id_csv"]))
log.info("KB selesai: EN=%d, ID=%d",len(kb_en),len(kb_id))
