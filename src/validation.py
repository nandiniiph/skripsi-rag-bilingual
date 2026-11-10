import json

SCENARIOS = ['S1_BM25', 'S2_DENSE', 'S3_HYBRID_RRF_RERANK']

def validate_inputs(qa, en, ind):
    for lang, kb in [('en', en), ('id', ind)]:
        needed = {'doc_id','content_id','parent_id','language','content'}
        if not needed.issubset(kb.columns):
            raise ValueError(f'Kolom KB {lang} kurang: {needed-set(kb.columns)}')
        if len(kb) < 50 or kb.doc_id.duplicated().any() or kb.content_id.duplicated().any():
            raise ValueError(f'KB {lang} perlu >=50 chunk dengan ID unik.')
        if not kb.language.eq(lang).all() or kb.content.isna().any() or kb.content.astype(str).str.strip().eq('').any():
            raise ValueError(f'KB {lang}: bahasa atau konten tidak valid.')
    if set(en.content_id) != set(ind.content_id):
        raise ValueError('content_id korpus EN dan ID tidak berpadanan.')
    if qa.qa_id.duplicated().any() or qa.question.isna().any() or qa.question.astype(str).str.strip().eq('').any():
        raise ValueError('qa_id duplikat / pertanyaan kosong.')
    if qa.reference_answer.isna().any() or qa.reference_answer.astype(str).str.strip().eq('').any():
        raise ValueError('Jawaban referensi kosong.')
    for pair_id, g in qa.groupby('pair_id'):
        if len(g) != 2 or set(g.language) != {'en', 'id'}:
            raise ValueError(f'Pasangan bilingual tidak lengkap: {pair_id}')
    ids = {'en': set(en.doc_id), 'id': set(ind.doc_id)}
    for q in qa.itertuples(index=False):
        rel = json.loads(q.relevant_doc_ids)
        if not isinstance(rel, list) or not rel or len(rel) != len(set(rel)) or not set(rel).issubset(ids[q.language]):
            raise ValueError(f'Ground truth tidak valid: {q.qa_id}')
