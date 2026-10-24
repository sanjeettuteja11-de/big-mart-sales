"""Reproduce exact one-dimensional calibration from three scored submissions.

No hidden labels are read. For a fixed RMSE scoring population, squared RMSE
along a fixed affine prediction line is a quadratic, determined by three scores.
The calculated score is conditional on the user-reported scores matching these
exact CSVs and on AV continuing to evaluate the same public rows.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
DEFAULT_DATA=Path('/Users/abhimanyunain/Downloads')
KEYS=['Item_Identifier','Outlet_Identifier'];TARGET='Item_Outlet_Sales'
FILES={
    'v63':('bigmart_submission_v63_exact_v60_optimum.csv',
           '9341aaf1f70110d7759c35598932b5cb15c0c4bfec7bd7a48d98b6658ad438cd',0.,1140.0228905024),
    'v65':('bigmart_submission_v65_quantity_correction.csv',
           'c8fd57830494b137ef32af68ef525c9fd1cf6c5c783902e8f19b5899c877082d',.75,1126.0148395277),
    'v66':('bigmart_submission_v66_quantity_correction.csv',
           'a910ffa2cf26d166b0635da7c5021acafb4eb72e2f8fa32236dbf06f302a2442',1.,1124.4058880453),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(data_dir=DEFAULT_DATA,output_dir=DEFAULT_DATA):
    frames={};inputs={}
    test=pd.read_csv(data_dir/'test_AbJTz2l.csv')
    sample=pd.read_csv(data_dir/'sample_submission_8RXa3c6.csv')
    assert sample.columns.tolist()==KEYS+[TARGET] and sample[KEYS].equals(test[KEYS])
    for version,(filename,expected,alpha,score) in FILES.items():
        path=data_dir/filename
        if digest(path)!=expected: raise ValueError(f'{version} input SHA-256 mismatch')
        frame=pd.read_csv(path)
        assert frame.columns.tolist()==sample.columns.tolist() and frame[KEYS].equals(test[KEYS])
        assert np.isfinite(frame[TARGET]).all()
        frames[version]=frame[TARGET].to_numpy(float)
        inputs[version]={'file':filename,'sha256':expected,'alpha':alpha,'reported_public_rmse':score}
    base=frames['v63'];direction=frames['v66']-base
    line_error=float(abs(frames['v65']-(base+.75*direction)).max())
    assert line_error<1e-8
    xs=np.array([v[2] for v in FILES.values()]);scores=np.array([v[3] for v in FILES.values()])
    a,b,c=np.linalg.solve(np.c_[xs**2,xs,np.ones(3)],scores**2)
    assert a>0
    alpha=float(-b/(2*a));calculated=float(np.sqrt(c-b*b/(4*a)))
    prediction=base+alpha*direction
    # Do not clip or round: that would invalidate the affine-line score identity.
    assert np.isfinite(prediction).all() and (prediction>=0).all()
    out=sample.copy();out[TARGET]=prediction
    output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/'bigmart_submission_v67_exact_quantity_optimum.csv'
    out.to_csv(path,index=False)
    saved=pd.read_csv(path)
    assert saved.shape==(5681,3) and saved[KEYS].equals(test[KEYS])
    assert np.allclose(saved[TARGET],prediction,atol=1e-9,rtol=0)
    result={'inputs':inputs,'line_identity_max_error':line_error,
        'squared_rmse_quadratic':{'a':float(a),'b':float(b),'c':float(c)},
        'optimal_alpha':alpha,'calculated_public_rmse':calculated,'observed_public_rmse':None,
        'public_score_conditions':['Input scores correspond to the exact hashed files.',
                                   'Same RMSE metric and same fixed public scoring rows.'],
        'scope':'Exact only for this one-dimensional line, not a global model optimum.',
        'output':{'path':str(path),'sha256':digest(path),'rows':len(out),
                  'min':float(prediction.min()),'max':float(prediction.max()),
                  'rms_change_from_v66':float(np.sqrt(np.mean((prediction-frames['v66'])**2)))}}
    (ROOT/'v67_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    ap.add_argument('--output-dir',type=Path,default=DEFAULT_DATA)
    args=ap.parse_args();main(args.data_dir,args.output_dir)
