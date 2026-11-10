import _bootstrap
import pandas as pd
from src.common import load_config, resolve_path, seed_everything
from src.generation import SailorGenerator, run_generation
cfg=load_config(); seed_everything(cfg['project']['seed']); e=cfg['experiment']; p=cfg['paths']
qa=pd.read_csv(resolve_path(p['qa_evaluation_csv']))
r=pd.read_csv(resolve_path(p['retrieval_dir'])/'retrieval_results.csv')
en=pd.read_csv(resolve_path(p['kb_en_csv'])); ind=pd.read_csv(resolve_path(p['kb_id_csv']))
g=SailorGenerator(cfg['models']['generator'])
out=run_generation(qa,r,en,ind,g,e['top_k_evidence'],e['max_new_tokens'],resolve_path(p['generation_csv']))
print(out.groupby(['scenario','language','status']).size().to_string())
print('Jalankan ulang perintah yang sama untuk melanjutkan/mencoba ulang technical_error.')
