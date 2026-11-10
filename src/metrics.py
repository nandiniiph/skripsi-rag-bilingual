from __future__ import annotations
import json
import re
import pandas as pd

RETRIEVAL_METRICS = ['Recall@1','Recall@3','Recall@5','MRR@50','Precision@3']
GENERATION_METRICS = ['rougeL_f1','bertscore_precision','bertscore_recall','bertscore_f1']


def parse_ids(value):
    return value if isinstance(value, list) else json.loads(value)


def recall_at_k(retrieved, relevant, k):
    rel = set(relevant)
    return len(set(retrieved[:k]) & rel) / len(rel) if rel else 0.0


def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & set(relevant)) / k if k > 0 else 0.0


def reciprocal_rank(retrieved, relevant, k=50):
    rel = set(relevant)
    for rank, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(qa_eval, retrieval):
    if qa_eval.qa_id.duplicated().any():
        raise ValueError('qa_id duplikat dalam data uji.')
    if retrieval.empty:
        raise ValueError('Hasil retrieval kosong.')
    if not set(retrieval.qa_id).issubset(set(qa_eval.qa_id)):
        raise ValueError('Hasil retrieval berisi qa_id di luar data uji.')
    scenarios = sorted(retrieval.scenario.unique())
    groups = {(q,s): g for (q,s),g in retrieval.groupby(['qa_id','scenario'])}
    rows = []
    for q in qa_eval.itertuples(index=False):
        relevant = parse_ids(q.relevant_doc_ids)
        if not relevant:
            raise ValueError(f'Ground truth kosong: {q.qa_id}')
        for scenario in scenarios:
            key = (q.qa_id, scenario)
            if key not in groups:
                raise ValueError(f'Hasil retrieval hilang: {key}. Bukan skor nol otomatis.')
            group = groups[key].sort_values('rank')
            if group.doc_id.duplicated().any() or group['rank'].tolist() != list(range(1,len(group)+1)):
                raise ValueError(f'Ranking/ID duplikat atau tidak berurutan: {key}')
            if not group.language.eq(q.language).all():
                raise ValueError('language harus tetap label sebenarnya, bukan hasil deteksi.')
            docs = group.doc_id.tolist()[:50]
            row = {'qa_id':q.qa_id, 'pair_id':q.pair_id, 'language':q.language, 'scenario':scenario,
                   'relevant_chunk_count':len(set(relevant)), 'retrieved_count':len(docs)}
            if 'predicted_language' in group:
                if group.predicted_language.nunique() != 1:
                    raise ValueError(f'Routing tidak konsisten: {key}')
                row['predicted_language'] = group.predicted_language.iloc[0]
                row['detection_correct'] = row['predicted_language'] == q.language
            row.update({f'Recall@{k}':recall_at_k(docs,relevant,k) for k in [1,3,5]})
            row['MRR@50'] = reciprocal_rank(docs,relevant,50)
            row['Precision@3'] = precision_at_k(docs,relevant,3)
            rows.append(row)
    per = pd.DataFrame(rows)
    summary = per.groupby(['scenario','language'],as_index=False)[RETRIEVAL_METRICS].mean()
    return per, summary


class UnicodeTokenizer:
    def tokenize(self, text):
        # Unicode letters/digits; underscore is a separator. No stemming.
        return re.findall(r'[^\W_]+', str(text).lower(), flags=re.UNICODE)


def complete_case_mask(frame, scenarios):
    complete = frame.assign(_ok=frame.status.eq('ok')).groupby('qa_id').agg(
        n=('scenario','nunique'), ok=('_ok','all'))
    ids = complete.index[(complete.n == len(scenarios)) & complete.ok]
    return frame.qa_id.isin(ids)


def evaluate_generation(generated, bert_model):
    from rouge_score import rouge_scorer
    from bert_score import BERTScorer
    g = generated.copy()
    if g.duplicated(['qa_id','scenario']).any():
        raise ValueError('Duplikasi keluaran qa_id/scenario.')
    if 'status' not in g:
        raise ValueError('Kolom status wajib. Gunakan hasil generasi revisi.')
    for col in ['generated_answer','reference_answer']:
        g[col] = g[col].fillna('').astype(str)
    if g.reference_answer.str.strip().eq('').any():
        raise ValueError('Referensi kosong tidak dapat dievaluasi.')
    g['empty_answer'] = g.generated_answer.str.strip().eq('') & g.status.eq('ok')
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, tokenizer=UnicodeTokenizer())
    for col in ['rougeL_precision','rougeL_recall'] + GENERATION_METRICS:
        g[col] = float('nan')
    for i,row in g.iterrows():
        if row.status != 'ok':
            continue
        if row.empty_answer:
            g.loc[i,['rougeL_precision','rougeL_recall']+GENERATION_METRICS] = 0.0
        else:
            score = scorer.score(row.reference_answer,row.generated_answer)['rougeL']
            g.loc[i,['rougeL_precision','rougeL_recall','rougeL_f1']] = [score.precision,score.recall,score.fmeasure]
    valid = g.index[g.status.eq('ok') & ~g.empty_answer]
    g['bertscore_candidate_truncated'] = False
    g['bertscore_reference_truncated'] = False
    if len(valid):
        bs = BERTScorer(model_type=bert_model,num_layers=9,idf=False,rescale_with_baseline=False)
        # Use the evaluator's own tokenizer, log text exceeding its supported input.
        tokenizer = bs._tokenizer
        limits = [x for x in [tokenizer.model_max_length] if isinstance(x,int) and 0 < x < 1000000]
        if not limits:
            raise ValueError('Batas tokenizer BERTScore tidak tersedia.')
        limit = min(limits)
        for i in valid:
            g.loc[i,'bertscore_candidate_truncated'] = len(tokenizer.encode(g.at[i,'generated_answer'],add_special_tokens=True,truncation=False)) > limit
            g.loc[i,'bertscore_reference_truncated'] = len(tokenizer.encode(g.at[i,'reference_answer'],add_special_tokens=True,truncation=False)) > limit
        P,R,F = bs.score(g.loc[valid,'generated_answer'].tolist(),g.loc[valid,'reference_answer'].tolist(),batch_size=8)
        for col,values in zip(['bertscore_precision','bertscore_recall','bertscore_f1'],[P,R,F]):
            g.loc[valid,col] = values.cpu().numpy()
        g['bertscore_hash'] = bs.hash
        g['bertscore_max_tokens'] = limit
    else:
        g['bertscore_hash'] = 'no_nonempty_successful_answers'
    from .validation import SCENARIOS
    if set(g.scenario) != set(SCENARIOS):
        raise ValueError('Hasil harus mencakup ketiga skenario utama.')
    g['included_in_comparison'] = complete_case_mask(g,SCENARIOS)
    summary = g[g.included_in_comparison].groupby(['scenario','language'],as_index=False)[GENERATION_METRICS].mean()
    counts = g.groupby(['scenario','language'],as_index=False).agg(
        n_total=('qa_id','size'), n_empty=('empty_answer','sum'), n_compared=('included_in_comparison','sum'),
        n_technical_errors=('status',lambda x:int(x.ne('ok').sum())))
    return g, counts.merge(summary,on=['scenario','language'],how='left')
