import numpy as np
from scipy.stats import wilcoxon,rankdata


def holm_correction(p_values):
    p=np.asarray(p_values,dtype=float)
    if not np.isfinite(p).all(): raise ValueError('p-value harus finite.')
    order=np.argsort(p);out=np.empty(len(p));previous=0.0
    for pos,i in enumerate(order):
        previous=max(previous,float(p[i])*(len(p)-pos))
        out[i]=min(previous,1.0)
    return out


def paired_wilcoxon(baseline,comparison):
    a=np.asarray(baseline,dtype=float);b=np.asarray(comparison,dtype=float)
    if len(a)!=len(b) or len(a)==0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Pasangan uji kosong/tidak valid.')
    # Same differences and zero rule for the test and effect size.
    d=np.round(b-a,12);nz=d[d!=0]
    if len(nz)==0:stat,p,effect=0.0,1.0,0.0
    else:
        result=wilcoxon(d,zero_method='wilcox',alternative='two-sided',method='auto')
        stat,p=float(result.statistic),float(result.pvalue)
        ranks=rankdata(abs(nz));effect=float((ranks[nz>0].sum()-ranks[nz<0].sum())/ranks.sum())
    return {'wilcoxon_statistic':stat,'p_value':p,'rank_biserial':effect,
            'improved':int((d>0).sum()),'same':int((d==0).sum()),'worse':int((d<0).sum())}
