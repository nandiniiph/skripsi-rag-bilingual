# Skripsi Nandini — Hybrid Sparse-Dense RAG Bilingual

Kode ini mengikuti metodologi penelitian BAB 3: MedQuAD → preprocessing → chunking → bilingual EN/ID → knowledge base → 300 QA evaluation → ground truth manual → BM25 / Multilingual-E5+FAISS → RRF → Top-3 → Sailor2 → Recall/MRR/ROUGE-L/BERTScore.

## 1. Dataset yang sudah disertakan
`data/raw/medquad.csv` adalah CSV MedQuAD yang diberikan pengguna, dengan kolom `question, answer, source, focus_area`.

## 2. Instalasi
Disarankan Python 3.11 dan virtual environment.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux
source .venv/bin/activate
pip install -r requirements.txt
```

Jika server memiliki GPU NVIDIA, instal build PyTorch/CUDA yang sesuai terlebih dahulu. `faiss-cpu` di requirements aman untuk CPU; ganti dengan build FAISS GPU bila environment mendukung.

## 3. Tahap data (3.4.2–3.4.6)
Jalankan satu per satu agar mudah memeriksa output:

```bash
python scripts/01_analyze_dataset.py
python scripts/02_preprocess.py
python scripts/03_chunk.py
python scripts/04_translate.py
```

Setelah translation, periksa manual:
- `data/manual_review/translation_auto_issues.csv`
- `data/manual_review/translation_review_200_chunks.csv`

Lanjut:
```bash
python scripts/05_build_kb.py
python scripts/06_sample_evaluation.py
python scripts/07_prepare_ground_truth.py
```

Script terakhir membuat `data/manual_review/ground_truth_annotation.csv`.

### Ground truth WAJIB manual
Untuk setiap kandidat chunk, isi `is_relevant = 1` hanya jika chunk secara langsung mengandung informasi yang diperlukan untuk menjawab `question`. Isi `0` untuk yang tidak relevan. Setiap `pair_id` harus memiliki minimal satu chunk relevan.

Setelah selesai:
```bash
python scripts/07_prepare_ground_truth.py --apply
```

Output utama:
- `data/evaluation/qa_evaluation.csv` (600 query: 300 EN + 300 ID)
- `data/manual_review/final_evaluation_translation_review.csv` untuk validasi bilingual final.

## 4. Retrieval dan evaluasi
```bash
python scripts/08_run_retrieval.py
python scripts/09_evaluate_retrieval.py
```

Konfigurasi default:
- BM25 Top-50
- Dense Top-50
- `intfloat/multilingual-e5-base`
- embedding L2-normalized
- FAISS `IndexFlatIP` (inner product = cosine similarity setelah normalisasi)
- RRF `k=60`
- Recall@1, Recall@3, Recall@5, MRR

## 5. Generasi jawaban
```bash
python scripts/10_run_generation.py
python scripts/11_evaluate_generation.py
```

Generator tetap: `sail/Sailor2-1B-Chat`, Top-3 evidence, `max_new_tokens=256`, `do_sample=False`.

Output akhir:
`outputs/evaluation/final_results.csv`

## 6. Parameter penelitian
Semua parameter berada di `config/config.yaml`. Ubah hanya jika metodologi skripsi juga direvisi agar naskah dan kode selalu sama.

## 7. Urutan output penting
1. `outputs/evaluation/dataset_analysis.json`
2. `data/processed/qa_pairs_en.csv`
3. `data/processed/chunks_en.csv`
4. `data/bilingual/qa_pairs_id.csv`
5. `data/bilingual/chunks_id.csv`
6. `data/knowledge_base/knowledge_base_en.csv`
7. `data/knowledge_base/knowledge_base_id.csv`
8. `data/evaluation/qa_sample_en.csv`
9. `data/manual_review/ground_truth_annotation.csv`
10. `data/evaluation/qa_evaluation.csv`
11. `outputs/retrieval/retrieval_results.csv`
12. `outputs/evaluation/retrieval_summary.csv`
13. `outputs/generation/generated_answers.csv`
14. `outputs/evaluation/final_results.csv`

## Catatan metodologis
- Normalisasi lowercase hanya untuk deteksi duplikat; teks asli yang sudah dibersihkan tetap dipakai untuk retrieval/generation.
- Tidak semua chunk dari satu jawaban otomatis relevan. Ground truth ditentukan manual.
- Query ID hanya mencari KB ID; query EN hanya mencari KB EN.
- Top-3 evidence berbeda dari Recall@3.
- Tidak ada fine-tuning Sailor2 atau Multilingual-E5.
