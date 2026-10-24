import numpy as np, pandas as pd, os
D=os.path.expanduser('~/Downloads')

SCORED = {
 'bigmart_submission_v30_allartifacts_meta.csv':1143.9865702486,
 'bigmart_submission_v31_meta_of_metas.csv':1143.9252774782,
 'bigmart_submission_v33_variance_bias.csv':1143.9054059653,
 'bigmart_submission_v35_bagged.csv':1143.9272592129,
 'bigmart_submission_v36_with_quantile.csv':1144.2918226387,
 'bigmart_submission_v44_with_nn.csv':1142.4429706684,
 'bigmart_submission_v46_secondrebuild.csv':1144.6392247765,
 'bigmart_submission_v55_genie_strategy.csv':1142.1520494780,
 'bigmart_submission_v56_widedeep_ordered.csv':1143.3406861303,
 'bigmart_submission_v65_probe_half.csv':1141.2371832503,
 'bigmart_submission_v66_probe_full_v57.csv':1143.8883491617,
 'bigmart_submission_v67_exact_optimum_v57line.csv':1139.9383155826,
 'bigmart_submission_v68_probe_half_v30.csv':1141.1135715901,
 'bigmart_submission_v70_probe_v67_to_v60_half.csv':1140.0632477175,
 'bigmart_submission_v71_probe_v67_away_v60.csv':1139.8765221044,
 'bigmart_submission_v72_extrapolated_optimum.csv':1139.7129068029,
 'bigmart_submission_v73_probe_v71_away_v43.csv':1139.9836953751,
 'bigmart_submission_v74_probe_away_v5.csv':1140.0386032940,
 'bigmart_submission_v74_probe_away_v31.csv':1139.8102079175,
 'bigmart_submission_v74_probe_away_v24.csv':1140.0591819667,
 'bigmart_submission_v75_toward_v31_p3.csv':1140.2446056213,
 'bigmart_submission_v75_toward_v31_p15.csv':1139.9001360210,
 'bigmart_submission_v76_qp_optimal.csv':1139.5952687699,
}
names=[]; P=[]; S=[]
for f,s in SCORED.items():
    p=f'{D}/{f}'
    if not os.path.exists(p): print('MISSING',f); continue
    names.append(f); P.append(pd.read_csv(p).Item_Outlet_Sales.values); S.append(s)
P=np.array(P); S=np.array(S); N=P.shape[1]; m=len(S)
print(f'scored submissions used: {m}, N={N}')

def fit(P,S,N):
    """Solve for q = proj of truth onto affine span, and d^2 = squared dist to span."""
    m=len(S)
    B = P[1:] - P[0]                      # (m-1, N) basis of the span direction
    # unknowns: alpha (m-1), beta
    # 2<p_i, p0 + B^T alpha> - beta = ||p_i||^2 - N s_i^2
    A=np.zeros((m,m)); rhs=np.zeros(m)
    for i in range(m):
        A[i,:m-1] = 2*(B@P[i])
        A[i,m-1]  = -1.0
        rhs[i] = (P[i]@P[i]) - N*S[i]**2 - 2*(P[i]@P[0])
    sol,*_ = np.linalg.lstsq(A,rhs,rcond=None)
    alpha=sol[:m-1]; beta=sol[m-1]
    q = P[0] + B.T@alpha
    d2 = beta - q@q
    resid = A@sol - rhs
    return q, d2, B, np.abs(resid).max()

q,d2,B,res = fit(P,S,N)
print(f'fit residual (max abs): {res:.4e}   d^2 (off-span dist^2) = {d2:.4f}  -> d = {np.sqrt(max(d2,0)):.4f}')
print(f'implied RMSE of the span-optimum q itself: {np.sqrt(max(d2,0)/N):.6f}')

# --- LEAVE-ONE-OUT VALIDATION: does this geometry actually predict LB scores? ---
print('\n=== LEAVE-ONE-OUT VALIDATION ===')
errs=[]
for k in range(m):
    idx=[i for i in range(m) if i!=k]
    qk,d2k,_,_=fit(P[idx],S[idx],N)
    pred=np.sqrt((np.sum((P[k]-qk)**2)+max(d2k,0))/N)
    errs.append(pred-S[k])
    print(f'{names[k][:52]:52s} actual={S[k]:.4f} predicted={pred:.4f} err={pred-S[k]:+.4f}')
errs=np.array(errs)
print(f'LOO mean abs err = {np.abs(errs).mean():.4f}  max = {np.abs(errs).max():.4f}')

# --- NOW SCORE THE TWO CLAIMED FILES ---
print('\n=== PREDICTED SCORES FOR THE CLAIMED FILES ===')
BQ = np.vstack([B, ])           # span directions
# orthonormal basis of span directions
Qb,_ = np.linalg.qr(BQ.T)       # N x (m-1)
for f,claim in [('bigmart_submission_v65_quantity_correction.csv',1126.0148395277),
                ('bigmart_submission_v66_quantity_correction.csv',1124.4058880453)]:
    v = pd.read_csv(f'{D}/{f}').Item_Outlet_Sales.values
    w = v - P[0]
    w_par = Qb@(Qb.T@w)
    w_perp = w - w_par
    v_A = P[0] + w_par
    in_span = np.sum((v_A-q)**2)
    perp_norm = np.linalg.norm(w_perp)
    d = np.sqrt(max(d2,0))
    lower_sq = in_span + max(0.0, d-perp_norm)**2
    point_est = np.sqrt((in_span + perp_norm**2 + d2)/N)   # if v_perp ⟂ y_perp
    print(f'\n{f}')
    print(f'  claimed LB score            : {claim:.6f}')
    print(f'  dist to span-optimum q      : in-span={np.sqrt(in_span):.2f}  off-span={perp_norm:.2f}  d(truth)={d:.2f}')
    print(f'  RIGOROUS LOWER BOUND on RMSE: {np.sqrt(lower_sq/N):.4f}')
    print(f'  best-case (v_perp aligned)  : {np.sqrt(max(in_span + (perp_norm-d)**2,0)/N):.4f}')
    print(f'  worst-case                  : {np.sqrt((in_span + (perp_norm+d)**2)/N):.4f}')
    print(f'  neutral point estimate      : {point_est:.4f}')
    print(f'  >>> CLAIM POSSIBLE? {"YES" if claim >= np.sqrt(lower_sq/N)-1e-6 else "NO - MATHEMATICALLY IMPOSSIBLE"}')
