import _bootstrap
import json
import re
import pandas as pd
from src.common import load_config,resolve_path
from src.language import detect_language

cfg=load_config(); p=cfg['paths']
out=resolve_path(p['retrieval_summary_csv']).parent; out.mkdir(parents=True,exist_ok=True)
qa=pd.read_csv(resolve_path(p['qa_evaluation_csv']))
det=pd.read_csv(resolve_path(p['retrieval_dir'])/'language_detection.csv')
if det.qa_id.duplicated().any() or set(det.qa_id)!=set(qa.qa_id):
    raise ValueError('Deteksi harus satu kali untuk setiap pertanyaan uji.')
truth=qa.set_index('qa_id').language
expected_questions=qa.set_index('qa_id').question
if not det.set_index('qa_id').question.eq(expected_questions.reindex(det.qa_id)).all():
    raise ValueError('Teks pertanyaan berubah sejak deteksi.')
if not det.set_index('qa_id').language.eq(truth.reindex(det.qa_id)).all():
    raise ValueError('Label bahasa deteksi berbeda dari data uji.')
det['correct']=det.language.eq(det.predicted_language)
det.groupby('language').agg(n=('qa_id','size'),accuracy=('correct','mean')).to_csv(out/'language_detection_summary.csv')
pd.crosstab(det.language,det.predicted_language).reindex(index=['id','en'],columns=['id','en'],fill_value=0).fillna(0).astype(int).to_csv(out/'language_confusion_matrix.csv')
(out/'language_detection_overall.json').write_text(json.dumps({'n':len(det),'accuracy':float(det.correct.mean())},indent=2))
g=pd.read_csv(resolve_path(p['generation_csv']),keep_default_na=False)
rows=[]
for r in g.to_dict('records'):
    text=r['generated_answer'].strip()
    success=r['status']=='ok'
    pred,score=detect_language(text) if success and text else ('',None)
    row={k:r[k] for k in ['qa_id','pair_id','scenario','language','predicted_language','generated_answer','status']}
    row.update(answer_language=pred,answer_language_score=score,
               nonempty_success=success and bool(text),empty_answer=success and not bool(text),
               language_match=(pred==r['language']) if pred else None)
    # Triage only; not an automatic judgement that an answer is medically correct.
    row['review_reason']=''
    if pred and pred!=r['language']:row['review_reason']='language_mismatch'
    if pred and len(re.findall(r'[^\W_]+',text))<=3:
        row['review_reason']+=';short_or_term_answer'
    rows.append(row)
per=pd.DataFrame(rows);per.to_csv(out/'answer_language_per_query.csv',index=False)
summary=[]
for (scenario,lang),sub in per.groupby(['scenario','language']):
    valid=sub[sub.nonempty_success]
    summary.append({'scenario':scenario,'language':lang,'n_total':len(sub),
        'n_nonempty':len(valid),'n_empty':int(sub.empty_answer.sum()),
        'n_technical_errors':int(sub.status.ne('ok').sum()),
        'language_match_rate':valid.language_match.astype(float).mean() if len(valid) else None})
pd.DataFrame(summary).to_csv(out/'answer_language_summary.csv',index=False)
review=per[per.review_reason.ne('')].copy()
for c in ['manual_language','mixed_language','notes']:review[c]=''
# Do not overwrite human annotations on a rerun.
review_path=out/'answer_language_manual_review.csv'
if not review_path.exists():review.to_csv(review_path,index=False)
else:review.to_csv(out/'answer_language_review_candidates.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
print('Pemeriksaan singkat <=3 kata hanya triase. Tambahkan kasus istilah/nama lain secara manual.')
