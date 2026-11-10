import _bootstrap  # noqa: F401
import pandas as pd
from src.common import load_config, resolve_path, setup_logging
from src.metrics import evaluate_generation
cfg=load_config(); log=setup_logging(resolve_path(cfg["paths"]["log_file"])); g=pd.read_csv(resolve_path(cfg["paths"]["generation_csv"]),keep_default_na=False); per,summary=evaluate_generation(g,cfg["models"]["bertscore"]); p=resolve_path(cfg["paths"]["generation_per_query_csv"]); p.parent.mkdir(parents=True,exist_ok=True); per.to_csv(p,index=False)
# gabungkan retrieval summary + generation summary menjadi final_results.csv
ret=pd.read_csv(resolve_path(cfg["paths"]["retrieval_summary_csv"])); final=ret.merge(summary,on=["scenario","language"],how="outer"); final.to_csv(resolve_path(cfg["paths"]["final_results_csv"]),index=False); print(final.to_string(index=False))

summary.to_csv(p.parent / "generation_summary.csv", index=False)
