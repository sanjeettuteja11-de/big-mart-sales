import numpy as np, pandas as pd, os
D=os.path.expanduser('~/Downloads')
exec(open(f'{D}/geometry_audit.py').read().split("q,d2,B,res = fit")[0])  # reuse SCORED, P,S,N,fit

q,d2,B,res = fit(P,S,N)
d=np.sqrt(max(d2,0))
Qb,_=np.linalg.qr(B.T)

print("="*70)
print("TEST 1: REQUIRED ALIGNMENT between the file's off-span part and truth's")
print("="*70)
for f,claim in [('bigmart_submission_v65_quantity_correction.csv',1126.0148395277),
                ('bigmart_submission_v66_quantity_correction.csv',1124.4058880453)]:
    v=pd.read_csv(f'{D}/{f}').Item_Outlet_Sales.values
    w=v-P[0]; wp=Qb@(Qb.T@w); wperp=w-wp
    vA=P[0]+wp
    insp=np.sum((vA-q)**2); n_perp=np.linalg.norm(wperp)
    # N*claim^2 = insp + n_perp^2 + d^2 - 2*n_perp*d*cos
    cos_req = (insp + n_perp**2 + d2 - N*claim**2)/(2*n_perp*d)
    print(f'\n{f}')
    print(f'  ||v_perp|| = {n_perp:,.0f}   ||y_perp|| = d = {d:,.0f}   ratio = {n_perp/d:.4f}')
    print(f'  REQUIRED cos(v_perp, y_perp) = {cos_req:.4f}   ({"IMPOSSIBLE >1" if cos_req>1 else "possible"})')
    print(f'  i.e. the file must capture {cos_req*n_perp:,.0f} of the {d:,.0f} unknown off-span signal')

print()
print("="*70)
print("TEST 2: what alignment do GENUINE models actually achieve? (holdout)")
print("="*70)
print("Refit span on 22 subs, treat the 23rd as an out-of-span 'new file', solve its cos.")
cosines=[]
for k in range(len(S)):
    idx=[i for i in range(len(S)) if i!=k]
    Pk=P[idx]; Sk=S[idx]
    qk,d2k,Bk,_=fit(Pk,Sk,N); dk=np.sqrt(max(d2k,0))
    Qk,_=np.linalg.qr(Bk.T)
    w=P[k]-Pk[0]; wp=Qk@(Qk.T@w); wperp=w-wp
    vA=Pk[0]+wp; insp=np.sum((vA-qk)**2); npr=np.linalg.norm(wperp)
    if npr<1e-6: continue
    c=(insp+npr**2+d2k-N*S[k]**2)/(2*npr*dk)
    cosines.append((names[k],npr,c))
    print(f'  {names[k][:50]:50s} ||perp||={npr:9.0f}  cos={c:+.4f}')
cs=np.array([c for _,_,c in cosines])
print(f'\n  genuine-model cos: mean={cs.mean():+.4f} std={cs.std():.4f} max={cs.max():+.4f} min={cs.min():+.4f}')

print()
print("="*70)
print("TEST 3: model-free triangle inequality vs every scored submission")
print("="*70)
for f,claim in [('bigmart_submission_v65_quantity_correction.csv',1126.0148395277),
                ('bigmart_submission_v66_quantity_correction.csv',1124.4058880453)]:
    v=pd.read_csv(f'{D}/{f}').Item_Outlet_Sales.values
    lo=0.0; who=''
    for i in range(len(S)):
        rms=np.sqrt(((v-P[i])**2).mean())
        b=S[i]-rms
        if b>lo: lo,who=b,f'{names[i]} (score {S[i]:.2f}, dist {rms:.1f})'
    print(f'{f}\n   tightest model-free lower bound = {lo:.2f}  from {who}')
    print(f'   claim {claim:.2f} -> {"consistent with triangle ineq" if claim>=lo else "VIOLATES triangle ineq"}')
