from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def _device_map():
    import torch
    return 'auto' if torch.cuda.is_available() else None


class SailorGenerator:
    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map=_device_map())
        if not torch.cuda.is_available():
            self.model.to('cpu')
        self.model.eval()
        limits = [getattr(self.model.config,'max_position_embeddings',None),self.tokenizer.model_max_length]
        limits = [int(n) for n in limits if n is not None and 0 < n < 1000000]
        if not limits:
            raise ValueError('Batas konteks generator tidak tersedia pada konfigurasi model.')
        self.context_limit = min(limits)
        self.last_metadata = {}

    def _prompt(self, question, evidence, language):
        ev = '\n\n'.join(f'[{i}] {x}' for i,x in enumerate(evidence,1))
        if language == 'id':
            user = f'Jawab dalam bahasa Indonesia hanya berdasarkan evidence yang diberikan. Jika evidence tidak cukup, nyatakan informasi yang tersedia tidak cukup.\n\nEvidence:\n{ev}\n\nPertanyaan:\n{question}'
        elif language == 'en':
            user = f'Answer in English using only the provided evidence. If the evidence is insufficient, state that the available information is insufficient.\n\nEvidence:\n{ev}\n\nQuestion:\n{question}'
        else:
            raise ValueError('Bahasa generator harus id/en.')
        if getattr(self.tokenizer,'chat_template',None):
            return self.tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
        return user+'\n\nAnswer:'

    def _encode(self, prompt, **kwargs):
        return self.tokenizer(prompt,add_special_tokens=not bool(getattr(self.tokenizer,'chat_template',None)),**kwargs)

    def prepare_context(self, question, evidence, language, max_new_tokens):
        budget = self.context_limit - max_new_tokens
        kept = list(evidence)
        original = len(self._encode(self._prompt(question,kept,language))['input_ids'])
        # Remove lower-ranked chunks first. Instruction and question stay intact.
        while True:
            prompt = self._prompt(question,kept,language)
            length = len(self._encode(prompt)['input_ids'])
            if length <= budget:
                break
            if not kept:
                raise ValueError('Pertanyaan dan instruksi melebihi anggaran konteks.')
            kept.pop()
        self.last_metadata = {'context_limit':self.context_limit,'prompt_tokens':length,
            'original_prompt_tokens':original,'context_truncated':len(kept)!=len(evidence),
            'evidence_used_count':len(kept),'evidence_dropped_count':len(evidence)-len(kept)}
        return prompt, kept

    def answer(self, question, evidence, language, max_new_tokens=256):
        import torch
        self.last_metadata = {}
        prompt,_ = self.prepare_context(question,evidence,language,max_new_tokens)
        inputs = self._encode(prompt,return_tensors='pt')
        device = next(self.model.parameters()).device
        inputs = {k:v.to(device) for k,v in inputs.items()}
        with torch.inference_mode():
            out = self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()


def _atomic_csv(frame,path):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(temporary,index=False)
    temporary.replace(path)


def run_generation(qa_eval,retrieval,kb_en,kb_id,generator,top_k=3,max_new_tokens=256,output_path=None):
    from .validation import SCENARIOS
    if qa_eval.qa_id.duplicated().any():
        raise ValueError('qa_id duplikat.')
    required = {'predicted_language','language','qa_id','scenario','rank','doc_id'}
    if not required.issubset(retrieval.columns):
        raise ValueError('Hasil retrieval belum memakai format deteksi bahasa revisi.')
    groups = {(q,s):g.sort_values('rank') for (q,s),g in retrieval.groupby(['qa_id','scenario'])}
    expected = {(q,s) for q in qa_eval.qa_id for s in SCENARIOS}
    if set(groups) != expected:
        raise ValueError('Kombinasi qa_id/skenario retrieval tidak lengkap.')
    kb = {'en':kb_en.set_index('doc_id'),'id':kb_id.set_index('doc_id')}
    fingerprint = hashlib.sha256()
    for frame in [qa_eval,retrieval,kb_en,kb_id]:
        fingerprint.update(pd.util.hash_pandas_object(frame,index=False).values.tobytes())
    fingerprint.update(Path(__file__).read_bytes())
    config_path = Path(__file__).resolve().parents[1] / 'config/config.yaml'
    if config_path.exists(): fingerprint.update(config_path.read_bytes())
    import importlib.metadata
    for name in ['torch', 'transformers', 'accelerate', 'pandas']:
        try: fingerprint.update((name+importlib.metadata.version(name)).encode())
        except importlib.metadata.PackageNotFoundError: pass
    model_revision = getattr(getattr(getattr(generator,'model',None),'config',None),'_commit_hash',None)
    fingerprint.update(json.dumps([getattr(generator,'model_name','test'),model_revision,top_k,max_new_tokens]).encode())
    signature = fingerprint.hexdigest()
    cached = {}
    if output_path:
        output_path = Path(output_path)
        meta_path = output_path.with_suffix('.manifest.json')
        if output_path.exists():
            if not meta_path.exists() or json.loads(meta_path.read_text())['signature'] != signature:
                raise ValueError('Checkpoint berbeda data/model/kode. Pilih folder output eksperimen baru.')
            old = pd.read_csv(output_path,keep_default_na=False)
            if old.duplicated(['qa_id','scenario']).any():
                raise ValueError('Checkpoint duplikat.')
            cached = {(r['qa_id'],r['scenario']):r for r in old.to_dict('records') if r['status']=='ok'}
        else:
            meta_path.parent.mkdir(parents=True,exist_ok=True)
            meta_path.write_text(json.dumps({'signature':signature,'model_revision':model_revision},indent=2))
    rows = dict(cached)
    for q in tqdm(qa_eval.itertuples(index=False),total=len(qa_eval),desc='Generation'):
        predictions = set()
        for scenario in SCENARIOS:
            key = (q.qa_id,scenario)
            group = groups[key]
            if not group.language.eq(q.language).all() or group.predicted_language.nunique()!=1:
                raise ValueError(f'Label/routing tidak konsisten: {key}')
            predicted = group.predicted_language.iloc[0]
            predictions.add(predicted)
            if len(predictions)>1:
                raise ValueError('Ketiga skenario harus menggunakan hasil deteksi yang sama.')
            if key in cached:
                continue
            docs = group.head(top_k).doc_id.tolist()
            if len(docs)!=top_k or len(set(docs))!=len(docs):
                raise ValueError(f'Evidence harus {top_k} chunk unik: {key}')
            evidence = [str(kb[predicted].loc[d,'content']) for d in docs]
            row = {'qa_id':q.qa_id,'pair_id':q.pair_id,'language':q.language,
                'predicted_language':predicted,'detection_correct':predicted==q.language,
                'scenario':scenario,'question':q.question,'reference_answer':q.reference_answer,
                'retrieved_doc_ids':json.dumps(docs),'retrieved_evidence':json.dumps(evidence,ensure_ascii=False)}
            try:
                answer = generator.answer(q.question,evidence,predicted,max_new_tokens=max_new_tokens)
                used = generator.last_metadata.get('evidence_used_count',len(evidence))
                row.update(status='ok',error='',generated_answer=answer,
                           used_evidence_doc_ids=json.dumps(docs[:used]),**generator.last_metadata)
            except Exception as exc:
                row.update(status='technical_error',error=f'{type(exc).__name__}: {exc}',generated_answer='')
                try:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except ImportError:
                    pass
            rows[key] = row
            if output_path:
                _atomic_csv(pd.DataFrame(rows.values()),output_path)
    return pd.DataFrame(rows.values())
