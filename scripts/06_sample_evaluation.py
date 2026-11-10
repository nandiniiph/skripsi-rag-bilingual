import _bootstrap  # noqa: F401
import pandas as pd
from src.common import load_config, resolve_path, setup_logging
from src.evaluation_dataset import sample_evaluation_pairs, save_sample
cfg=load_config(); log=setup_logging(resolve_path(cfg["paths"]["log_file"]))
qa=pd.read_csv(resolve_path(cfg["paths"]["cleaned_qa_en_csv"]))
out=sample_evaluation_pairs(qa,cfg["experiment"]["evaluation_pairs"],cfg["project"]["seed"]); save_sample(out,resolve_path(cfg["paths"]["qa_sample_en_csv"]))
log.info("Sampel evaluation dibuat: %d pasangan EN",len(out))
