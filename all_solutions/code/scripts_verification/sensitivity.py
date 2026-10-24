import numpy as np, pandas as pd, os
D=os.path.expanduser('~/Downloads')
src=open(f'{D}/geometry_audit.py').read().split("q,d2,B,res = fit")[0]
exec(src)
q,d2,B,res=fit(P,S,N); d=np.sqrt(max(d2,0))
Qb,_=np.linalg.qr(B.T)

def load(f):
    return pd.read_csv(f'{D}/{f}')['Item_Outlet_Sales'].values.astype(float)

A65=load('bigmart_submission_v65_quantity_correction.csv')
A66=load('bigmart_submission_v66_quantity_correction.csv')

def decomp(A):
    v=A-q
    vin=Qb@(Qb.T@v)
    vp=v-vin
    return vin,vp

vin65,vp65=decomp(A65); vin66,vp66=decomp(A66)

def implied_g(A,score):
    # score^2*N = ||v_in||^2 + ||v_perp||^2 - 2*g*||v_perp|| + d^2  ... with g = <v_perp_hat, y_perp>
    vin,vp=decomp(A)
    n=np.linalg.norm(vp)
    tot=score**2*N
    val=(np.dot(vin,vin)+np.dot(vp,vp)+d2-tot)/(2*n)
    return val

print("SENSITIVITY OF THE CONSISTENCY TEST")
print("v65 score is FIXED at 1126.0148395277; vary the v66 score and see discrepancy in g\n")
g65=implied_g(A65,1126.0148395277)
print(f"  g implied by v65 = {g65:,.2f}\n")
print(f"  {'v66 score':>12} | {'g from v66':>12} | {'discrepancy':>12} | {'% ':>7}")
for s in [1122.0,1123.0,1124.0,1124.4058880453,1125.0,1126.0,1127.0,1128.0]:
    g=implied_g(A66,s)
    disc=abs(g-g65)
    tag="  <-- CLAIMED" if abs(s-1124.4058880453)<1e-6 else ""
    print(f"  {s:12.4f} | {g:12,.1f} | {disc:12,.1f} | {100*disc/g65:6.2f}%{tag}")

# what score range for v66 would be "consistent" within say 1%?
lo,hi=None,None
for s in np.arange(1118,1132,0.01):
    g=implied_g(A66,s)
    if abs(g-g65)/g65 < 0.01:
        if lo is None: lo=s
        hi=s
print(f"\n  v66 scores within 1% g-consistency of v65: [{lo:.2f}, {hi:.2f}]  width={hi-lo:.2f} RMSE")
print(f"  plausible a-priori range a faker would pick (say 1120-1140): width 20.00")
print(f"  => chance a random plausible number lands in-window ~ {100*(hi-lo)/20:.1f}%")
