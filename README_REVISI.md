# Revisi kode eksperimen: deteksi bahasa otomatis

Paket ini memperbarui kode yang dikirim pada 19 Oktober 2026. Tidak berisi dataset,
model, indeks atau hasil eksperimen. Proposal tidak diubah.

## Pemasangan pada server

1. Unggah arsip revisi ke `/home/nandini` melalui VS Code Remote SSH.
2. Ekstrak paket dan pasang kode. Installer mencadangkan setiap file yang diubah,
   menolak konflik jika kode server berubah sejak arsip awal, dan tidak menyentuh
   dataset, indeks atau hasil lama.

```bash
cd ~
tar -xzf kode_skripsi_revisi.tar.gz
python3 skripsi_revisi/INSTALL.py --target "$HOME/skripsi_baru"
cd ~/skripsi_baru
source env/bin/activate
python -m pip install langid==1.1.6
python -m pip check
python -m unittest discover -s tests -v
python scripts/08_check_inputs.py
```

Gunakan environment eksperimen lama yang sudah berjalan. `requirements.txt`
memuat langid dan scipy secara eksplisit; jangan melakukan upgrade semua pustaka
hanya untuk memasang revisi. Jika ada dependency belum terpasang, pasang sesuai
requirements. Pengujian ringan memerlukan numpy, pandas, scipy, PyYAML,
rouge-score dan langid. Uji ini tidak mengunduh E5/BGE/Sailor/BERTScore.

## Urutan eksperimen setelah pemeriksaan input lolos

Jalankan satu per satu, periksa hasil setiap tahap:

```bash
python scripts/08_run_retrieval.py
python scripts/09_evaluate_retrieval.py
python scripts/09a_compare_s3_reranking.py
python scripts/10_run_generation.py
python scripts/11_evaluate_generation.py
python scripts/11b_evaluate_language.py
python scripts/11a_analyze_bilingual.py
python scripts/12_statistical_test.py
```

Semua keluaran eksperimen baru berada di `outputs/auto_language_v1/`.
Data uji, knowledge base, dan anotasi lama tetap menjadi input. Tidak perlu
mengulang preprocessing, penerjemahan, sampling, atau anotasi jika input lolos
pemeriksaan. Pemeriksaan struktur tidak membuktikan kebenaran medis terjemahan.
Jika pemeriksaan gagal, baca pesan dan perbaiki masalah data yang disebutkan;
jangan melewati pemeriksaan dengan mengisi semua anotasi 1.

Retrieval menolak menimpa `retrieval_results.csv` yang sudah ada. Untuk eksperimen
baru dengan perubahan data/konfigurasi, gunakan folder output baru secara
konsisten pada seluruh `paths` keluaran di config. Hasil sebelumnya tetap disimpan.

Generasi menyimpan checkpoint per jawaban. Jalankan ulang `10_run_generation.py`
untuk melanjutkan dan mencoba kembali baris `technical_error` setelah penyebabnya
diperbaiki. Checkpoint menolak perubahan data, kode generator, konfigurasi, versi
pustaka utama atau revisi model. Jika perubahan diperlukan, gunakan output baru.
Jawaban kosong yang selesai diproses berstatus `ok` dan diberi skor nol.
Kegagalan teknis tidak diberi skor nol; dilaporkan terpisah. Perbandingan generasi
menggunakan pertanyaan yang lengkap pada ketiga skenario.

## Demo aplikasi

```bash
python scripts/12_build_demo_indexes.py
python app.py
```

Indeks demo baru berada di `indexes/auto_language_v1/`. Aplikasi memeriksa
urutan doc_id, jumlah vektor dan hash isi KB sebelum menggunakan indeks.
Bahasa dikenali otomatis; pilihan bahasa manual sudah dihapus.

## Perubahan utama

- langid 1.1.6 dibatasi pada ID/EN. Deteksi dilakukan sekali per pertanyaan,
  digunakan bersama oleh S1/S2/S3. Tidak ada penerjemahan query saat inferensi.
- `language` tetap label sebenarnya untuk pengelompokan evaluasi.
  `predicted_language` menentukan korpus dan instruksi bahasa generator.
  Kesalahan deteksi tetap masuk evaluasi. Skor langid bukan ukuran akurasi
  terkalibrasi dan tidak dipakai sebagai ambang penolakan.
- Top-50 setiap skenario disimpan. MRR@50 hanya menilai posisi 1–50.
  Precision@3 selalu menggunakan penyebut 3.
- BM25 eksplisit k1=1.5, b=0.75, epsilon=0.25, mempertahankan default
  BM25Okapi pada kode asli. Tidak mengganti menjadi k1=1.2.
- RRF k=60, Top-10 BGE, batch 8, max_length 512, Top-3 evidence tetap.
  Skor BGE dan skor RRF disimpan terpisah; urutkan dengan `rank`, bukan `score`
  karena kolom kompatibilitas `score` mencampur dua jenis skor pada S3.
- BGE memakai truncation longest-first bawaan untuk pasangan pertanyaan–chunk;
  panjang sebelum pemotongan dan status truncation dicatat pada kandidat BGE.
- E5 max_seq_length 512, passage/query prefix dan normalisasi tetap.
  `e5_token_audit.csv` mencatat panjang masukan dan potensi truncation.
  Revisi ini tidak otomatis membagi ulang chunk terjemahan; jika ada yang
  melebihi batas, tinjau data sebelum mengklaim seluruh chunk masuk utuh.
- Generator memakai instruksi bahasa eksplisit, greedy decoding (do_sample=False,
  num_beams=1), dan max_new_tokens=256. Batas konteks dibaca dari model/tokenizer.
  Jika terlalu panjang, chunk terendah dilepas terlebih dahulu. Pertanyaan dan
  instruksi tetap utuh. `used_evidence_doc_ids` mencatat chunk yang benar-benar
  masuk prompt. Recall/Precision tetap menilai ranking sebelum pengurangan.
- ROUGE-L memakai kata/angka Unicode, lowercase, tanpa stemming.
  BERTScore xlm-roberta-base, lapisan 9, IDF=False, rescaling=False.
  Pemotongan oleh tokenizer BERTScore ditandai; hasil tetap metrik referensi,
  bukan validasi klinis. Evaluator menyimpan hash konfigurasi BERTScore.
- Ground truth kosong/belum lengkap ditolak. Pemeriksaan input mengecek seluruh
  kandidat chunk sumber, kesesuaian qrels, pasangan bilingual, dan penerapan
  koreksi terjemahan final pada data uji/KB. Cakupan relevansi masih jawaban
  sumber, bukan anotasi seluruh korpus.
- Statistik utama: tiga pasangan skenario per bahasa/metrik, Holm pada tiga uji
  tersebut. Ablasi reranking menjadi keluarga terpisah satu uji per bahasa/metrik.
  Selisih dibulatkan 12 desimal, zero_method=wilcox, rank-biserial mengikuti
  aturan nol yang sama. Unit statistik adalah jawaban sumber: pertanyaan dengan
  jawaban sumber identik setelah normalisasi dikelompokkan dan skornya dirata-rata.
  Ini tidak membuktikan independensi antarjawaban berbeda dari halaman web sama.
- Anotasi dan review yang sudah ada tidak ditimpa otomatis saat membentuk ulang
  ground truth. URL `source` dipertahankan bila preprocessing kelak diulang.
  Revisi tidak mengembalikan URL yang sudah hilang pada CSV lama.

## Pemeriksaan bahasa jawaban

`11b_evaluate_language.py` menghasilkan akurasi dan confusion matrix bahasa
pertanyaan, kesesuaian bahasa jawaban, serta kandidat pemeriksaan manual.
Penyebut kesesuaian bahasa adalah jawaban sukses yang tidak kosong. Jawaban
kosong dan technical_error dilaporkan terpisah. Kandidat manual mencakup
mismatch dan keluaran <=3 kata sebagai triase; tambahkan kasus nama/istilah lain
secara manual bila diperlukan. Flag otomatis tidak sama dengan validasi manusia.
Hasil otomatis tidak ditimpa oleh anotasi manual.

## Berkas penting

- evaluation/experiment_manifest.json: konfigurasi, hash input/kode, versi,
  platform dan GPU yang tersedia.
- retrieval/language_detection.csv: satu prediksi per pertanyaan.
- retrieval/model_revisions.json: revisi model embedding/reranker.
- retrieval/retrieval_results.csv dan s3_rrf_before_rerank.csv.
- generation/generated_answers.csv dan generated_answers.manifest.json.
- evaluation/final_results.csv: ringkasan retrieval dan jawaban.
- evaluation/language_detection_summary.csv dan language_confusion_matrix.csv.
- evaluation/answer_language_summary.csv dan answer_language_manual_review.csv.
- evaluation/statistical_tests.csv dan statistical_source_units.csv.

## Batas validasi paket

Sintaks dan pengujian logika dijalankan pada lingkungan audit. langid dan
ROUGE dijalankan dengan pustaka nyata; inferensi E5, BGE, Sailor2, dan BERTScore
menggunakan pengganti terkontrol dalam pengujian logika. Dataset serta GPU server
pengguna tidak tersedia pada audit. Karena itu paket belum dinyatakan lolos
inferensi end-to-end pada 600 pertanyaan. Simpan pesan kesalahan jika uji server
gagal. Skrip eksplorasi 08a/08b/08c tetap disertakan sebagai kode lama, bukan
bagian urutan eksperimen utama revisi.

Referensi implementasi:
https://github.com/saffsd/langid.py
https://github.com/Tiiiger/bert_score
https://github.com/google-research/google-research/tree/master/rouge
