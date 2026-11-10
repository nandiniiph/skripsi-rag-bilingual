import _bootstrap  # noqa: F401
import pandas as pd
from src.common import load_config, resolve_path, setup_logging
from src.translation import NLLBTranslator, translate_qa_file, translate_chunks_file, build_translation_quality_files
cfg=load_config(); log=setup_logging(resolve_path(cfg["paths"]["log_file"])); t=cfg["translation"]
translator=NLLBTranslator(cfg["models"]["translation"],t["source_lang"],t["target_lang"])
qa_id=translate_qa_file(resolve_path(cfg["paths"]["cleaned_qa_en_csv"]),resolve_path(cfg["paths"]["qa_id_csv"]),translator,t["batch_size"],t["max_input_tokens"])
chunks_id=translate_chunks_file(resolve_path(cfg["paths"]["chunks_en_csv"]),resolve_path(cfg["paths"]["chunks_id_csv"]),translator,t["batch_size"],t["max_input_tokens"])
chunks_en=pd.read_csv(resolve_path(cfg["paths"]["chunks_en_csv"]))
build_translation_quality_files(chunks_en,chunks_id,resolve_path(cfg["paths"]["translation_issues_csv"]),resolve_path(cfg["paths"]["translation_review_csv"]),cfg["preprocessing"]["translation_review_sample"],cfg["project"]["seed"])
log.info("Translation selesai. Lakukan review manual file %s",cfg["paths"]["translation_review_csv"])
