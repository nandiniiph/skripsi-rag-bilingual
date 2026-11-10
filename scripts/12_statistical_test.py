import _bootstrap
from itertools import combinations
import hashlib
import pandas as pd
from src.common import load_config,resolve_path
from src.metrics import RETRIEVAL_METRICS,GENERATION_METRICS
from src.statistics import paired_wilcoxon,holm_correction
from src.validation import SCENARIOS

cfg=load_config();p=cfg['paths'];out=resolve_path(p['final_results_csv']).parent
qa=pd.read_csv(resolve_path(p['qa_evaluation_csv']))
sample=pd.read_csv(resolve_path(p['qa_sample_en_csv']))
# Unit = jawaban sumber. Identical normalized answers share a unit, even if parent_id differs.
# Not a claim of independence between different answers from the same webpage.
if sample.pair_id.duplicated().any():raise ValueError('pair_id duplikat pada sampel.')
sample['source_unit']=sample.answer.map(lambda x:hashlib.sha256(' '.join(str(x).casefold().split()).encode()).hexdigest())
units=qa[['qa_id','pair_id']].merge(sample[['pair_id','source_unit']],on='pair_id',validate='many_to_one')
if len(units)!=len(qa):raise ValueError('Pemetaan sumber data uji tidak lengkap.')
units.to_csv(out/'statistical_source_units.csv',index=False)
rows=[]

def analyze(frame,metrics,names,analysis):
    if frame.duplicated(['qa_id','scenario']).any():raise ValueError('Skor duplikat.')
    frame=frame.merge(units[['qa_id','source_unit']],on='qa_id',validate='many_to_one')
    for lang in ['en','id']:
        data=frame[frame.language.eq(lang)]
        # Keep exactly the same complete queries across metrics and scenarios in this analysis.
        tables=[data.pivot(index='qa_id',columns='scenario',values=m).reindex(columns=names) for m in metrics]
        complete=set(tables[0].dropna().index)
        for t in tables[1:]:complete &= set(t.dropna().index)
        data=data[data.qa_id.isin(complete)]
        aggregate=data.groupby(['source_unit','scenario'])[metrics].mean().reset_index()
        for metric in metrics:
            wide=aggregate.pivot(index='source_unit',columns='scenario',values=metric).reindex(columns=names).dropna()
            if len(wide)<2:
                raise ValueError(f'Kurang dari dua unit sumber lengkap: {analysis}/{lang}/{metric}')
            family=[]
            for a,b in combinations(names,2):
                result=paired_wilcoxon(wide[a],wide[b])
                family.append({'analysis':analysis,'language':lang,'metric':metric,
                    'comparison':f'{b} vs {a}','baseline':a,'comparison_scenario':b,
                    'n_queries':len(complete),'n_source_units':len(wide),
                    'mean_baseline':wide[a].mean(),'mean_comparison':wide[b].mean(),
                    'mean_difference':(wide[b]-wide[a]).mean(),**result})
            adjusted=holm_correction([x['p_value'] for x in family])
            for row,p_adj in zip(family,adjusted):
                row.update(p_value_holm=p_adj,significant_0_05=p_adj<.05,
                    holm_family=f'{analysis}/{lang}/{metric}',family_size=len(family))
            rows.extend(family)

ret=pd.read_csv(resolve_path(p['retrieval_per_query_csv']))
gen=pd.read_csv(resolve_path(p['generation_per_query_csv']))
gen=gen[gen.included_in_comparison.astype(str).str.lower().eq('true')]
analyze(ret,RETRIEVAL_METRICS,SCENARIOS,'main_retrieval')
analyze(gen,['rougeL_f1','bertscore_f1'],SCENARIOS,'main_generation')
ablation=pd.read_csv(out/'s3_reranking_comparison_per_query.csv')
analyze(ablation,RETRIEVAL_METRICS,['S3_RRF_BEFORE_RERANK',SCENARIOS[2]],'reranking')
results=pd.DataFrame(rows);results.to_csv(out/'statistical_tests.csv',index=False)
print(results.to_string(index=False))
print('Holm: tiga pasangan per metrik/bahasa untuk analisis utama. Ablasi keluarga terpisah (satu uji).')
print('Rank-biserial positif: skenario comparison lebih tinggi. Pembulatan selisih 12 desimal; zero_method=wilcox.')
