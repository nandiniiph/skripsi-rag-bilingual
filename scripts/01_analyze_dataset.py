import _bootstrap
import pandas as pd
from src.common import load_config, resolve_path, setup_logging
from src.dataset_analysis import analyze_dataset
cfg=load_config(); log=setup_logging(resolve_path(cfg["paths"]["log_file"]))
df=pd.read_csv(resolve_path(cfg["paths"]["raw_csv"]))
report=analyze_dataset(df,resolve_path(cfg["paths"]["dataset_analysis_json"]),resolve_path(cfg["paths"]["focus_distribution_csv"]),resolve_path(cfg["paths"]["artifact_candidates_csv"]))
log.info("Analisis selesai: %s", report)
