import numpy as np, pandas as pd, os
D=os.path.expanduser('~/Downloads')
src=open(f'{D}/geometry_audit.py').read().split("q,d2,B,res = fit")[0]
exec(src)
q,d2,B,res=fit(P,S,N); d=np.sqrt(max(d2,0))
Qb,_=np.linalg.qr(B.T)

def load(f):
    return pd.read_csv(f'{D}/{f}')['Item_Outlet_Sales'].values.astype(float)

files = {
 'v65 (mine)': ('bigmart_submission_v65_quantity_correction.csv', 1126.0148395277),
 'v66 (mine)': ('bigmart_submission_v66_quantity_correction.csv', 1124.4058880453),
 'v79 (mine, computed)': ('bigmart_submission_v79_quantity_line_optimum.csv', None),
 'v68 (codex)': ('bigmart_submission_v68_denoised_quantity.csv', 1122.8091692185),
 'v69 (codex)': ('bigmart_submission_v69_denoised_quantity.csv', 1122.9470444204),
}

vecs={}
for name,(f,s) in files.items():
    a=load(f)
    v=a-q
    vin=Qb@(Qb.T@v)
    vp=v-vin
    vecs[name]=(a,v,vin,vp,s)
    print(f"{name:25s} ||v_perp||={np.linalg.norm(vp):10.1f}  claimed={s}")

print("\n=== PAIRWISE COSINE of v_perp directions ===")
names=list(vecs.keys())
for i in range(len(names)):
    for j in range(i+1,len(names)):
        a=vecs[names[i]][3]; b=vecs[names[j]][3]
        na=np.linalg.norm(a); nb=np.linalg.norm(b)
        if na<1e-6 or nb<1e-6: continue
        cos=np.dot(a,b)/(na*nb)
        print(f"  {names[i]:25s} vs {names[j]:25s}  cos={cos:8.5f}")
