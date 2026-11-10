"""Deteksi ID/EN. Tidak menerima label ground truth sebagai masukan."""
from functools import lru_cache
from threading import Lock

@lru_cache(maxsize=1)
def _identifier():
    from langid.langid import LanguageIdentifier, model
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    identifier.set_languages(['id', 'en'])
    return identifier

_LOCK = Lock()

def detect_language(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Teks untuk deteksi bahasa harus berupa string tidak kosong.')
    with _LOCK:
        language, score = _identifier().classify(text.strip())
    return language, float(score)


def detect_questions(qa):
    if qa['qa_id'].duplicated().any():
        raise ValueError('qa_id duplikat.')
    if not qa['language'].isin(['id', 'en']).all():
        raise ValueError('Label bahasa sebenarnya harus id/en.')
    rows = []
    for q in qa.itertuples(index=False):
        predicted, score = detect_language(q.question)
        rows.append({'qa_id': q.qa_id, 'pair_id': q.pair_id,
                     'language': q.language, 'question': q.question,
                     'predicted_language': predicted, 'language_score': score,
                     'detection_correct': predicted == q.language})
    import pandas as pd
    return pd.DataFrame(rows)
