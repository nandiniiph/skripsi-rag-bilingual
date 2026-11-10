import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.metrics import evaluate_generation
from src.validation import SCENARIOS

class Array:
    def __init__(self,n):self.n=n
    def cpu(self):return self
    def numpy(self):return np.full(self.n,.7)
class FakeBERTScorer:
    def __init__(self,**kw):
        assert kw['num_layers']==9 and kw['idf'] is False and kw['rescale_with_baseline'] is False
        self._tokenizer=types.SimpleNamespace(model_max_length=512,encode=lambda text,**kw:list(range(len(text.split())+2)))
        self.hash='mock-test-only'
    def score(self,candidates,references,**kw):
        assert all(candidates)
        return [Array(len(candidates)) for _ in range(3)]

class MetricsTests(unittest.TestCase):
    def test_empty_zero_failure_excluded_all_scenarios(self):
        rows=[]
        for q in ['complete','failed']:
            for i,s in enumerate(SCENARIOS):
                rows.append({'qa_id':q,'scenario':s,'language':'id','status':'technical_error' if q=='failed' and i==0 else 'ok',
                    'generated_answer':'' if i==0 else 'café 12','reference_answer':'café 12'})
        module=types.SimpleNamespace(BERTScorer=FakeBERTScorer)
        with patch.dict(sys.modules,{'bert_score':module}):
            per,summary=evaluate_generation(pd.DataFrame(rows),'xlm-roberta-base')
        self.assertEqual(per.loc[0,'rougeL_f1'],0)
        self.assertEqual(per.loc[0,'bertscore_f1'],0)
        self.assertEqual(per.loc[1,'rougeL_f1'],1)
        self.assertTrue(pd.isna(per.loc[3,'bertscore_f1']))
        self.assertTrue(per.loc[per.qa_id.eq('failed'),'included_in_comparison'].eq(False).all())
        self.assertTrue(summary.n_compared.eq(1).all())
