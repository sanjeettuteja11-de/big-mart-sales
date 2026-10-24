import numpy as np, pandas as pd, os
D=os.path.expanduser('~/Downloads')
exec(open(f'{D}/geometry_audit.py').read().split("q,d2,B,res = fit")[0])
q,d2,B,res=fit(P,S,N); d=np.sqrt(max(d2,0)); Qb,_=np.linalg.qr(B.T)

v65=pd.read_csv(f'{D}/bigmart_submission_v65_quantity_correction.csv').Item_Outlet_Sales.values
v66=pd.read_csv(f'{D}/bigmart_submission_v66_quantity_correction.csv').Item_Outlet_Sales.values
c65,c66=1126.0148395277,1124.4058880453

def decomp(v):
    w=v-P[0]; wp=Qb@(Qb.T@w); return P[0]+wp, w-wp
A65,p65=decomp(v65); A66,p66=decomp(v66)
cosperp=(p65@p66)/(np.linalg.norm(p65)*np.linalg.norm(p66))
print(f'cos(perp65, perp66) = {cosperp:.6f}  -> {"PARALLEL (same direction, diff magnitude)" if cosperp>0.99 else "NOT parallel - independent directions"}')
print(f'||p65||={np.linalg.norm(p65):.1f}  ||p66||={np.linalg.norm(p66):.1f}  ratio={np.linalg.norm(p66)/np.linalg.norm(p65):.4f}')
print(f'in-span dist to q: v65={np.linalg.norm(A65-q):.1f}  v66={np.linalg.norm(A66-q):.1f}  (nearly identical => same in-span point)')
print(f'cos(A65-q, A66-q) = {((A65-q)@(A66-q))/(np.linalg.norm(A65-q)*np.linalg.norm(A66-q)):.6f}')

print("\n--- If perps are parallel, the two scores are ONE constraint, not two.")
print("--- Test: does a SINGLE consistent truth explain BOTH exactly?")
# unknown scalar: projection of y_perp onto the shared perp direction, call it g
u = p65/np.linalg.norm(p65)
a65=np.linalg.norm(A65-q)**2; n65=np.linalg.norm(p65)
a66=np.linalg.norm(A66-q)**2; n66=np.linalg.norm(p66)
# N*s^2 = a + n^2 + d2 - 2*n*g   =>  g = (a+n^2+d2-N s^2)/(2n)
g65=(a65+n65**2+d2-N*c65**2)/(2*n65)
g66=(a66+n66**2+d2-N*c66**2)/(2*n66)
print(f'  implied g (truth component along shared dir) from v65 score: {g65:,.2f}')
print(f'  implied g from v66 score                                  : {g66:,.2f}')
print(f'  DISCREPANCY: {abs(g65-g66):,.2f}  ({abs(g65-g66)/max(abs(g65),1)*100:.3f}% )')
print(f'  -> {"CONSISTENT: one truth explains both scores" if abs(g65-g66)/max(abs(g65),1)<0.02 else "INCONSISTENT"}')

print("\n--- SANITY: is g <= d (Cauchy-Schwarz)?")
print(f'  g={g65:,.0f}  d={d:,.0f}  g/d={g65/d:.4f}  {"OK" if g65<=d else "VIOLATION - IMPOSSIBLE"}')

print("\n=== IMPLIED TRUE SCORE OF THE PURE CORRECTION DIRECTION ===")
# optimal step along u from A (the in-span point) : minimize a + t^2 - 2 t g  -> t=g
best_t=g65
best_rmse=np.sqrt((a65 + 0 - g65**2 + d2)/N)
print(f'  optimal magnitude along this direction: t* = {best_t:,.0f}')
print(f'  v65 used t={n65:,.0f}, v66 used t={n66:,.0f}')
print(f'  RMSE at optimal t*: {best_rmse:.4f}')
# also allow moving in-span to q
best2=np.sqrt((d2-g65**2)/N)
print(f'  RMSE if ALSO at in-span optimum q: {best2:.4f}')

print("\n=== EXTRAPOLATION: what does t beyond n66 give? ===")
for t in [n65,n66,1.5*n66,2*n66,best_t,1.2*best_t]:
    r=np.sqrt((a66*0 + np.linalg.norm(A66-q)**2 + t**2 - 2*t*g65 + d2)/N)
    print(f'   t={t:9,.0f} -> predicted RMSE {r:.4f}')
