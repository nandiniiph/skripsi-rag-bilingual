import _bootstrap  # noqa: F401
import pandas as pd
from src.common import load_config, resolve_path, setup_logging
from src.preprocessing import preprocess_dataframe, save_preprocessed
cfg=load_config(); log=setup_logging(resolve_path(cfg["paths"]["log_file"]))
df=pd.read_csv(resolve_path(cfg["paths"]["raw_csv"]))
out,report=preprocess_dataframe(df)
report_path=resolve_path("outputs/evaluation/preprocessing_report.json")
save_preprocessed(out,resolve_path(cfg["paths"]["cleaned_qa_en_csv"]),report_path,report)
log.info("Preprocessing selesai: %s",report)
