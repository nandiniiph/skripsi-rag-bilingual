import argparse, subprocess, sys
STAGES={
 "analyze":"scripts/01_analyze_dataset.py","preprocess":"scripts/02_preprocess.py","chunk":"scripts/03_chunk.py",
 "translate":"scripts/04_translate.py","kb":"scripts/05_build_kb.py","sample":"scripts/06_sample_evaluation.py",
 "groundtruth":"scripts/07_prepare_ground_truth.py","retrieval":"scripts/08_run_retrieval.py","eval_retrieval":"scripts/09_evaluate_retrieval.py",
 "generation":"scripts/10_run_generation.py","eval_generation":"scripts/11_evaluate_generation.py"}
p=argparse.ArgumentParser(); p.add_argument("stage",choices=list(STAGES)+["data"]); args=p.parse_args()
order=["analyze","preprocess","chunk","translate","kb","sample"] if args.stage=="data" else [args.stage]
for s in order:
    print(f"\n=== {s} ==="); subprocess.run([sys.executable,STAGES[s]],check=True)
if args.stage=="data":
    print("\nTahap otomatis data selesai. Berikutnya jalankan ground truth tanpa --apply, isi CSV manual, lalu jalankan --apply.")
