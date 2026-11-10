import _bootstrap
import json
import platform
import hashlib
import importlib.metadata
from pathlib import Path
import pandas as pd
from src.common import load_config,resolve_path,ROOT
from src.validation import validate_inputs

cfg=load_config();p=cfg['paths']
if cfg.get('language_detection',{}).get('candidate_languages')!=['id','en']:
    raise ValueError('Versi eksperimen ini mendukung kandidat ID/EN saja.')
if cfg['experiment']['bm25_top_n']!=50 or cfg['experiment']['dense_top_n']!=50 or cfg['evaluation']['mrr_cutoff']!=50:
    raise ValueError('Versi penelitian ini menetapkan kandidat 50 dan MRR@50.')
if cfg['evaluation']['bertscore_num_layers']!=9 or cfg['evaluation']['bertscore_idf'] is not False or cfg['evaluation']['bertscore_rescale_with_baseline'] is not False:
    raise ValueError('Evaluator versi ini memakai lapisan 9, tanpa IDF/rescaling.')
if cfg['experiment']['top_k_evidence']!=3 or cfg['experiment']['do_sample'] is not False:
    raise ValueError('Versi penelitian ini menggunakan Top-3 dan do_sample=False.')
qa=pd.read_csv(resolve_path(p['qa_evaluation_csv']));en=pd.read_csv(resolve_path(p['kb_en_csv']));ind=pd.read_csv(resolve_path(p['kb_id_csv']))
validate_inputs(qa,en,ind)
if len(qa)!=cfg['experiment']['evaluation_pairs']*2:raise ValueError('Jumlah query tidak sesuai konfigurasi.')
review_path=resolve_path(p['eval_translation_review_csv'])
if not review_path.exists():raise ValueError('File pemeriksaan terjemahan final belum tersedia.')
review=pd.read_csv(review_path)
status=pd.to_numeric(review.is_valid,errors='coerce')
if not status.eq(1).all():raise ValueError('Review final belum disetujui seluruhnya. Selesaikan pemeriksaan manual dahulu.')
if set(review.pair_id)!=set(qa.pair_id):raise ValueError('Pasangan review final berbeda dengan data uji.')
# Confirm review corrections were actually applied, not merely filled in a separate CSV.
questions=qa[qa.language.eq('id')].set_index('pair_id'); chunks=ind.set_index('content_id')
for r in review.itertuples(index=False):
    revision=getattr(r,'revision','')
    expected=str(revision).strip() if pd.notna(revision) and str(revision).strip() else str(r.indonesian_text).strip()
    if r.type in ['question','reference_answer']:
        actual=questions.loc[r.pair_id,r.type]
    elif r.type=='relevant_chunk':
        actual=chunks.loc[r.content_id,'content']
    else:raise ValueError(f'Tipe review tidak dikenal: {r.type}')
    if str(actual).strip()!=expected:raise ValueError(f'Revisi terjemahan belum diterapkan: {r.pair_id}/{r.type}')
for pair,g in review.groupby('pair_id'):
    if not {'question','reference_answer'}.issubset(set(g.type)):raise ValueError(f'Review tidak lengkap: {pair}')
    expected={d[:-3] for d in json.loads(questions.loc[pair,'relevant_doc_ids'])}
    if expected != set(g[g.type.eq('relevant_chunk')].content_id):raise ValueError(f'Review chunk relevan tidak lengkap: {pair}')
annotation=pd.read_csv(resolve_path(p['gt_annotation_csv']))
labels=pd.to_numeric(annotation.is_relevant,errors='coerce')
if labels.isna().any() or not labels.isin([0,1]).all():raise ValueError('Anotasi ground truth memiliki baris kosong/tidak valid.')
if annotation.duplicated(['pair_id','doc_id']).any():raise ValueError('Anotasi duplikat.')
sample=pd.read_csv(resolve_path(p['qa_sample_en_csv']))
if set(annotation.pair_id)!=set(sample.pair_id):raise ValueError('Pasangan anotasi berbeda dengan sampel.')
for q in sample.itertuples(index=False):
    a=annotation[annotation.pair_id.eq(q.pair_id)]
    expected=set(en.loc[en.parent_id.eq(q.parent_id),'doc_id'])
    if set(a.doc_id)!=expected:raise ValueError(f'Anotasi tidak mencakup semua chunk sumber: {q.pair_id}')
    relevant=set(a.loc[pd.to_numeric(a.is_relevant).eq(1),'doc_id'])
    row=qa[qa.pair_id.eq(q.pair_id) & qa.language.eq('en')].iloc[0]
    if set(json.loads(row.relevant_doc_ids))!=relevant:raise ValueError(f'Ground truth belum sesuai anotasi: {q.pair_id}')
counts=qa.assign(relevant_chunk_count=qa.relevant_doc_ids.map(lambda x:len(json.loads(x))))
out=resolve_path(p['retrieval_summary_csv']).parent;out.mkdir(parents=True,exist_ok=True)
counts[['qa_id','pair_id','language','relevant_chunk_count']].to_csv(out/'ground_truth_counts.csv',index=False)
versions={}
for name in ['torch','transformers','sentence-transformers','rank-bm25','faiss-cpu','langid','rouge-score','bert-score','pandas','numpy','scipy']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]='NOT_INSTALLED'
manifest={'config':cfg,'python':platform.python_version(),'platform':platform.platform(),'versions':versions,'inputs':{},'source_hashes':{}}
for key in ['qa_evaluation_csv','kb_en_csv','kb_id_csv','qa_sample_en_csv', 'eval_translation_review_csv']:
    path=resolve_path(p[key]);manifest['inputs'][key]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
for path in sorted((ROOT/'src').glob('*.py'))+sorted((ROOT/'scripts').glob('*.py')):
    manifest['source_hashes'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
try:
    import torch
    manifest['cuda_available']=torch.cuda.is_available()
    manifest['gpu_names']=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
except ImportError:manifest['cuda_available']=False
(out/'experiment_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print(f'Input valid: {len(qa)} query; EN {len(en)} chunk; ID {len(ind)} chunk.')
print('Pemeriksaan ini memeriksa struktur dan penerapan review, bukan menggantikan validasi makna medis.')
