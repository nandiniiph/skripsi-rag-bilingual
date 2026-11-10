import unittest
import tempfile
from pathlib import Path
import json
import ast
from unittest.mock import patch
import pandas as pd
import numpy as np
from src.metrics import recall_at_k,precision_at_k,reciprocal_rank,evaluate_retrieval,UnicodeTokenizer,complete_case_mask
from src.language import detect_language,detect_questions
from src.generation import run_generation,SailorGenerator
from src.ground_truth import build_qa_evaluation
from src.statistics import holm_correction,paired_wilcoxon
from src.validation import SCENARIOS


def fixture():
    en=pd.DataFrame([{'doc_id':f'd{i}-en','content_id':f'd{i}','parent_id':'p1','language':'en','content':f'English content {i}'} for i in range(1,61)])
    ind=en.copy();ind['doc_id']=ind.content_id+'-id';ind['language']='id';ind['content']='Konten Indonesia'
    qa=pd.DataFrame([{'qa_id':'q-en','pair_id':'q','language':'en','question':'What is diabetes?','reference_answer':'A condition.','relevant_doc_ids':'["d1-en"]'},
                     {'qa_id':'q-id','pair_id':'q','language':'id','question':'Apa itu diabetes?','reference_answer':'Suatu kondisi.','relevant_doc_ids':'["d1-id"]'}])
    rows=[]
    for q in qa.itertuples():
        for scenario in SCENARIOS:
            for rank in range(1,51):
                # Force the ID question to the wrong EN corpus.
                rows.append({'qa_id':q.qa_id,'pair_id':q.pair_id,'language':q.language,'predicted_language':'en','scenario':scenario,'doc_id':f'd{rank}-en','rank':rank})
    return qa,en,ind,pd.DataFrame(rows)


class FakeGenerator:
    model_name='fake'
    def __init__(self,fail_once=False):self.calls=[];self.fail_once=fail_once;self.last_metadata={}
    def answer(self,question,evidence,language,max_new_tokens):
        self.calls.append((question,evidence,language))
        if self.fail_once:self.fail_once=False;raise RuntimeError('simulated failure')
        self.last_metadata={'evidence_used_count':3}
        return '' if question.startswith('Apa') else 'English answer'


class RevisionTests(unittest.TestCase):
    def test_mrr_cutoff(self):
        docs=[f'd{i}' for i in range(1,101)]
        self.assertEqual(reciprocal_rank(docs,['d51']),0)
        self.assertEqual(reciprocal_rank(docs,['d50']),1/50)
    def test_precision_fixed_denominator(self):
        self.assertEqual(precision_at_k(['a'],['a'],3),1/3)
        self.assertEqual(precision_at_k([],['a'],3),0)
        self.assertEqual(recall_at_k(['a'],['a','b'],3),.5)
    def test_unicode_tokenizer(self):
        self.assertEqual(UnicodeTokenizer().tokenize('CAFÉ naïve 12 µg, sakit_kepala'),['café','naïve','12','µg','sakit','kepala'])
    def test_wrong_route_uses_gold_for_evaluation(self):
        qa,en,ind,r=fixture();per,_=evaluate_retrieval(qa,r)
        self.assertTrue(per.loc[per.language.eq('id'),'Recall@1'].eq(0).all())
        self.assertTrue(per.loc[per.language.eq('en'),'Recall@1'].eq(1).all())
    def test_missing_retrieval_raises(self):
        qa,en,ind,r=fixture()
        with self.assertRaises(ValueError):evaluate_retrieval(qa,r[~(r.qa_id.eq('q-id') & r.scenario.eq(SCENARIOS[0]))])
    def test_incomplete_annotation_raises(self):
        ann=pd.DataFrame({'pair_id':['q','q'],'doc_id':['d1-en','d2-en'],'is_relevant':[1,None]})
        with self.assertRaises(ValueError):build_qa_evaluation(pd.DataFrame(),pd.DataFrame(),ann)
    def test_language_real_model(self):
        self.assertEqual(detect_language('Apa saja gejala penyakit diabetes dan bagaimana cara mengatasinya?')[0],'id')
        self.assertEqual(detect_language('What are the symptoms of diabetes and how is it treated?')[0],'en')
        with self.assertRaises(ValueError):detect_language(' ')
    def test_question_detection_once(self):
        qa,*_=fixture()
        with patch('src.language.detect_language',return_value=('en',.9)) as d:
            result=detect_questions(qa)
            self.assertEqual(d.call_count,len(qa));self.assertEqual(result.detection_correct.tolist(),[True,False])
    def test_generation_wrong_route_and_resume(self):
        qa,en,ind,r=fixture();gen=FakeGenerator(fail_once=True)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'answers.csv'
            first=run_generation(qa,r,en,ind,gen,output_path=path)
            self.assertEqual(first.status.eq('technical_error').sum(),1)
            self.assertEqual(len(gen.calls),6)
            self.assertTrue(all(call[2]=='en' for call in gen.calls))
            self.assertTrue(all(call[1][0].startswith('English') for call in gen.calls))
            second=run_generation(qa,r,en,ind,gen,output_path=path)
            self.assertEqual(len(gen.calls),7)
            self.assertTrue(second.status.eq('ok').all())
            self.assertEqual(second.generated_answer.eq('').sum(),3)
            modified=qa.copy();modified.loc[0,'question']='Changed question'
            with self.assertRaises(ValueError):run_generation(modified,r,en,ind,gen,output_path=path)
    def test_complete_cases(self):
        frame=pd.DataFrame([{'qa_id':'q','scenario':s,'status':'ok' if i else 'technical_error'} for i,s in enumerate(SCENARIOS)])
        self.assertFalse(complete_case_mask(frame,SCENARIOS).any())
    def test_holm_and_effect(self):
        np.testing.assert_allclose(holm_correction([.01,.04,.03]),[.03,.06,.06])
        self.assertEqual(paired_wilcoxon([1,1],[1,1])['p_value'],1)
        self.assertEqual(paired_wilcoxon([0]*6,[1]*6)['rank_biserial'],1)
        self.assertEqual(paired_wilcoxon([1]*6,[0]*6)['rank_biserial'],-1)
    def test_context_trims_lowest_rank(self):
        generator=object.__new__(SailorGenerator)
        class Tokenizer:
            chat_template=None
            def __call__(self,text,**kwargs):return {'input_ids':text.split()}
        generator.tokenizer=Tokenizer();generator.context_limit=55
        prompt,kept=generator.prepare_context('What is diabetes?',['a '*10,'b '*10,'c '*10],'en',10)
        self.assertTrue(generator.last_metadata['context_truncated'])
        self.assertEqual(kept,['a '*10])
        self.assertIn('What is diabetes?',prompt)
        self.assertLessEqual(generator.last_metadata['prompt_tokens'],45)
    def test_rrf_and_rerank_pipeline_mocked(self):
        # Execute the actual retrieval function without downloading E5/BGE.
        path=Path(__file__).resolve().parents[1]/'src/retrieval.py'
        tree=ast.parse(path.read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['rrf','run_all_retrieval']]
        qa,en,ind,_=fixture()
        calls=[]
        class Retriever:
            def __init__(self,kb,**kwargs):self.kb=kb
            def search(self,q,top_n):
                calls.append(self.kb.language.iloc[0])
                return [{'doc_id':x,'rank':i,'score':1/i} for i,x in enumerate(self.kb.doc_id[:top_n],1)]
        class Reranker:
            def __init__(self,**kwargs):pass
            def rerank(self,query,candidates,doc_text):
                self_count=len(candidates)
                if self_count!=10:raise AssertionError(self_count)
                return [{'doc_id':c['doc_id'],'rrf_rank':c['rank'],'rrf_score':c['score'],'reranker_score':i} for i,c in enumerate(reversed(candidates))]
        from src.validation import validate_inputs
        from src.common import ensure_parent
        ns={'json':json,'pd':pd,'Path':Path,'SentenceTransformer':lambda x:None,'BM25Retriever':Retriever,
            'DenseRetriever':Retriever,'BGEReranker':Reranker,'ensure_parent':ensure_parent,
            'validate_inputs':validate_inputs,'detect_questions':lambda df:pd.DataFrame({'qa_id':df.qa_id,'predicted_language':['en','en']}),
            'tqdm':lambda x,**kw:x}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
        with tempfile.TemporaryDirectory() as d:
            result=ns['run_all_retrieval'](qa,en,ind,'mock','mock',50,50,60,10,Path(d))
            self.assertEqual(len(result),300)
            self.assertEqual(calls,['en']*4)
            self.assertTrue(result.groupby(['qa_id','scenario']).size().eq(50).all())
            hybrid=result[result.scenario.eq(SCENARIOS[2]) & result.qa_id.eq('q-id')]
            self.assertEqual(hybrid.doc_id.iloc[0],'d10-en')
            self.assertEqual(hybrid.doc_id.iloc[10],'d11-en')
            self.assertTrue(hybrid.language.eq('id').all())

if __name__=='__main__':unittest.main()
