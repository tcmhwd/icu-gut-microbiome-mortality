import pandas as pd, glob, re, os
B='/home/tcmhwd/microbiome/ICU_public'
m=pd.read_csv(f'{B}/semantic_microbiome/harmonized/sample_id_mapping.csv')
coh=pd.read_csv(f'{B}/semantic_microbiome/harmonized/discovery_cohort_primary.csv')
mm=m[(m.view=='MetaPhlAn')&m.patient_id_clinical.isin(coh.patient_id_clinical)]
want=dict(zip(mm.sample_id_in_view,mm.patient_id_clinical)); print('target samples',len(want))
best={}
for out in glob.glob(f'{B}/logs/bwa_map_*.out'):
    try: t=open(out,errors='ignore').read()
    except: continue
    r=re.search(r'R1=\S*/([^/\s]+)_host_removed',t)
    if not r or r.group(1) not in want or '[DONE] BAM ready' not in t: continue
    err=out[:-4]+'.err'
    if not os.path.exists(err): continue
    e=open(err,errors='ignore').read()
    if 'Real time' not in e: continue
    n=sum(int(x) for x in re.findall(r'Processed (\d+) reads',e))
    best.setdefault(r.group(1),[]).append((os.path.getmtime(err),n,out))
rows=[]
for s,v in best.items():
    v.sort(); ns=set(x[1] for x in v)
    rows.append((s,v[-1][1],len(v),len(ns)))
d=pd.DataFrame(rows,columns=['sample','reads','nlogs','ndistinct'])
print('found',len(d),'missing',len(set(want)-set(d['sample'])),'samples with disagreeing reruns',(d.ndistinct>1).sum())
pairs=d.reads/2
print('read pairs (M): median %.2f IQR %.2f-%.2f min %.2f max %.2f mean %.2f'%tuple(x/1e6 for x in [pairs.median(),pairs.quantile(.25),pairs.quantile(.75),pairs.min(),pairs.max(),pairs.mean()]))
print('reads total (M): median %.2f IQR %.2f-%.2f'%tuple(x/1e6 for x in [d.reads.median(),d.reads.quantile(.25),d.reads.quantile(.75)]))
print((pairs<1e6).sum(),'samples <1M pairs;',(pairs<5e6).sum(),'<5M')
