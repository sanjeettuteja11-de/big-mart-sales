import pandas as pd, numpy as np, os, glob

D = os.path.expanduser('~/Downloads')
train = pd.read_csv(f'{D}/train_v9rqX0R.csv')
test  = pd.read_csv(f'{D}/test_AbJTz2l.csv')
samp  = pd.read_csv(f'{D}/sample_submission_8RXa3c6.csv')
v65 = pd.read_csv(f'{D}/bigmart_submission_v65_quantity_correction.csv')
v66 = pd.read_csv(f'{D}/bigmart_submission_v66_quantity_correction.csv')

print("=== FORMAT ===")
for n,d in [('v65',v65),('v66',v66),('sample',samp)]:
    print(n, d.shape, list(d.columns), 'nan:', int(d.isna().sum().sum()),
          'neg:', int((d.Item_Outlet_Sales<0).sum()) if 'Item_Outlet_Sales' in d else '-')

# key alignment vs sample
k = lambda d: d.Item_Identifier+'|'+d.Outlet_Identifier
print("v65 keys == sample keys (order):", (k(v65)==k(samp)).all())
print("v66 keys == sample keys (order):", (k(v66)==k(samp)).all())
print("dup keys v65:", k(v65).duplicated().sum())

print("\n=== VALUE STATS ===")
for n,d in [('v65',v65),('v66',v66)]:
    s=d.Item_Outlet_Sales
    print(n, 'min %.3f max %.1f mean %.2f std %.2f'%(s.min(),s.max(),s.mean(),s.std()))
print('train target: min %.2f max %.1f mean %.2f std %.2f'%(
    train.Item_Outlet_Sales.min(),train.Item_Outlet_Sales.max(),
    train.Item_Outlet_Sales.mean(),train.Item_Outlet_Sales.std()))

print("\n=== v65 vs v66 ===")
d = v66.Item_Outlet_Sales.values - v65.Item_Outlet_Sales.values
print('corr %.8f'%np.corrcoef(v65.Item_Outlet_Sales,v66.Item_Outlet_Sales)[0,1])
print('mean abs diff %.3f  max abs diff %.2f  rms diff %.3f'%(np.abs(d).mean(),np.abs(d).max(),np.sqrt((d**2).mean())))

# ---- THE CLAIM: "quantity correction" ----
# hypothesis: Item_Outlet_Sales is an integer multiple of a per-item-per-outlet unit price
print("\n=== QUANTITY-STRUCTURE TEST ON TRAIN ===")
tr = train.dropna(subset=['Item_Outlet_Sales']).copy()
r = tr.Item_Outlet_Sales / tr.Item_MRP
print('sales/MRP: min %.4f max %.4f'%(r.min(),r.max()))
frac = np.abs(r - np.round(r))
print('fraction of train rows where sales/MRP is within 0.01 of an integer: %.4f'%(frac<0.01).mean())
print('within 0.05: %.4f'%(frac<0.05).mean())

# the real known structure: per (Item, Outlet) sales = qty * price, price unknown.
# Test: within each Outlet, is sales/MRP clustered on integers after a scale factor?
for out,g in tr.groupby('Outlet_Identifier'):
    rr = g.Item_Outlet_Sales/g.Item_MRP
    best=None
    for s in np.arange(0.80,1.30,0.001):
        f=np.abs(rr/s - np.round(rr/s))
        hit=(f<0.02).mean()
        if best is None or hit>best[1]: best=(s,hit)
    print(f'  {out}: n={len(g)} best scale={best[0]:.3f} integer-hit-rate={best[1]:.3f}')

# ---- Are v65/v66 themselves quantized? ----
print("\n=== ARE PREDICTIONS QUANTIZED? ===")
for n,d in [('v65',v65),('v66',v66)]:
    m = test.merge(d, on=['Item_Identifier','Outlet_Identifier'], how='left')
    rr = m.Item_Outlet_Sales/m.Item_MRP
    f = np.abs(rr-np.round(rr))
    print(n,'sales/MRP integer-hit-rate (<0.02): %.4f'%(f<0.02).mean(),
          ' decimals look continuous:', d.Item_Outlet_Sales.astype(str).str.split('.').str[1].str.len().median())

# ---- compare to prior submissions we produced ----
print("\n=== COMPARISON TO PRIOR SUBMISSIONS ===")
prev = sorted(glob.glob(f'{D}/bigmart_submission_v*.csv'))
base = None
for p in prev:
    name=os.path.basename(p)
    try:
        dd=pd.read_csv(p)
        if dd.shape[0]!=5681: continue
        c=np.corrcoef(dd.Item_Outlet_Sales,v66.Item_Outlet_Sales)[0,1]
        rms=np.sqrt(((dd.Item_Outlet_Sales.values-v66.Item_Outlet_Sales.values)**2).mean())
        print(f'{name:55s} corr_to_v66={c:.6f} rms_to_v66={rms:8.2f}')
    except Exception as e:
        print(name,'ERR',e)
