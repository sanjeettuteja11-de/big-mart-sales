"""Bounded continuation study; all predictions cached with configurations and fingerprints.

python -m src.continue_study --stage baseline
python -m src.continue_study --stage structural
"""
from __future__ import annotations
import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from src.artifacts import provenance, save_result
from src.config import OUTPUTS, TARGET
from src.cv import run_cv, rmse
from src.data import load_raw
from src.models import SPECS_BY_NAME, load_tuned

DEST = OUTPUTS / 'strict_study'
BOOSTERS = ['catboost_units_unweighted', 'catboost_units', 'catboost_raw',
            'xgboost_units_unweighted', 'xgboost_raw', 'lightgbm_raw']
BASELINES = ['store_rate_popularity', 'store_rate_plain', 'store_type_rate_popularity',
             'ridge_interactions', 'poisson_glm', *BOOSTERS, 'lightgbm_units_popularity']


def evaluate(name, train, test, tag, params=None, repeats=3, seed=42, scheme='stratified', drop=None):
    spec = SPECS_BY_NAME[name]
    params = load_tuned(name) if params is None else params
    config = {'model': name, 'params': {**spec.params, **params}, 'repeats': repeats,
              'seed': seed, 'scheme': scheme, 'drop': sorted(drop or []), 'fold_fit': True,
              'protocol': 'strict-inner-preprocessing-v1'}
    prov = provenance(train, test)
    path = DEST / tag
    if (path / 'result.json').exists():
        record = json.loads((path / 'result.json').read_text())
        if record['config'] == config and record['provenance'] == prov:
            return record, np.load(path / 'oof.npy'), np.load(path / 'test.npy')
    result = run_cv(spec, train, test, params=params, n_repeats=repeats, seed=seed,
                    scheme=scheme, drop=drop, fold_fit=True)
    record = save_result(result, path, config, prov)
    print(f'{tag}: {result.oof_rmse:.5f} RMSE ({result.seconds:.1f}s)', flush=True)
    return record, result.oof, result.test


def structural_grid():
    # 5 x 4 x 3 = 60 combinations, fixed before reading results.
    for rate in [0.0, 100.0, 500.0, 2000.0, float('inf')]:
        for smooth in [20.0, 55.0, 150.0, float('inf')]:
            for power in [0.0, 1.0, 2.0]:
                yield dict(rate_smoothing=rate, smoothing=smooth, weight_power=power)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['baseline','structural','confirm'], required=True)
    args = parser.parse_args()
    train, test = load_raw()
    if args.stage == 'baseline':
        rows = []
        for name in BASELINES:
            r, _, _ = evaluate(name, train, test, 'baseline/' + name)
            rows.append(r)
        (DEST/'baseline.json').write_text(json.dumps(rows, indent=2))
    elif args.stage == 'structural':
        rows = []
        for i,p in enumerate(structural_grid()):
            r,_,_ = evaluate('hierarchical_store_rate', train, test, f'screen/grid_{i:02}', p, repeats=1)
            rows.append(r)
        best = min(rows, key=lambda r:r['oof_rmse_full_precision'])
        for i,extra in enumerate([{'integer_units': True}, *[{'interaction_smoothing': v} for v in [100,500,2000]]]):
            p = {**best['config']['params'], **extra}
            r,_,_ = evaluate('hierarchical_store_rate', train, test, f'screen/refinement_{i}', p, repeats=1)
            rows.append(r)
        rows.sort(key=lambda r:r['oof_rmse_full_precision'])
        (DEST/'structural_screen.json').write_text(json.dumps(rows, indent=2))
        print('TOP 5', [(r['oof_rmse_full_precision'],r['config']['params']) for r in rows[:5]],flush=True)
    else:
        screen = json.loads((DEST/'structural_screen.json').read_text())
        params = [r['config']['params'] for r in screen[:3]]
        rows = []
        for i,p in enumerate(params):
            for seed in [42,137,2026]:
                r,_,_ = evaluate('hierarchical_store_rate',train,test,f'confirm/candidate_{i}_seed_{seed}',p,seed=seed)
                rows.append(r)
        for seed in [137,2026]:
            evaluate('store_rate_popularity',train,test,f'confirm/baseline_seed_{seed}',seed=seed)
        (DEST/'structural_confirm.json').write_text(json.dumps(rows,indent=2))

if __name__ == '__main__':
    main()
