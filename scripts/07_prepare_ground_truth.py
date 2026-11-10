import _bootstrap  # noqa: F401
import argparse, pandas as pd
from src.common import load_config, resolve_path, setup_logging
from src.ground_truth import create_annotation_candidates, save_annotation, build_qa_evaluation, create_final_translation_review
p=argparse.ArgumentParser(); p.add_argument("--apply",action="store_true"); args=p.parse_args()
cfg=load_config(); log=setup_logging(resolve_path(cfg["paths"]["log_file"]))
sample=pd.read_csv(resolve_path(cfg["paths"]["qa_sample_en_csv"])); kb_en=pd.read_csv(resolve_path(cfg["paths"]["kb_en_csv"])); kb_id=pd.read_csv(resolve_path(cfg["paths"]["kb_id_csv"])); qa_id=pd.read_csv(resolve_path(cfg["paths"]["qa_id_csv"]))
ann_path=resolve_path(cfg["paths"]["gt_annotation_csv"])
if not args.apply:
    if ann_path.exists(): raise FileExistsError("Anotasi sudah ada. Jangan menimpa hasil pemeriksaan manual.")
    ann=create_annotation_candidates(sample,kb_en); save_annotation(ann,ann_path)
    print(f"\nAnotasi manual wajib. Isi kolom is_relevant (1 untuk relevan) pada:\n{ann_path}\nSetelah selesai jalankan: python scripts/07_prepare_ground_truth.py --apply")
else:
    if resolve_path(cfg["paths"]["eval_translation_review_csv"]).exists(): raise FileExistsError("Review bilingual sudah ada. Arsipkan dengan sengaja sebelum membentuk ulang data uji.")
    ann=pd.read_csv(ann_path); qa_eval=build_qa_evaluation(sample,qa_id,ann); out=resolve_path(cfg["paths"]["qa_evaluation_csv"]); out.parent.mkdir(parents=True,exist_ok=True); qa_eval.to_csv(out,index=False)
    review=create_final_translation_review(qa_eval,kb_en,kb_id); rp=resolve_path(cfg["paths"]["eval_translation_review_csv"]); rp.parent.mkdir(parents=True,exist_ok=True); review.to_csv(rp,index=False)
    log.info("QA evaluation selesai: %d query. Lakukan validasi bilingual final di %s",len(qa_eval),rp)
