"""Auditable model results: data/source fingerprints, exact folds and predictions."""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import ROOT, OUTPUTS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def provenance(train, test):
    snapshots = OUTPUTS / 'source_snapshots'
    snapshots.mkdir(exist_ok=True)
    source = {p.name: sha256(p) for p in sorted((ROOT / 'src').glob('*.py'))
              if p.stem in ('cv', 'structure', 'data', 'features', 'models', 'targets', 'price_model', 'lattice_boost')}
    for name, digest in source.items():
        path = snapshots / (digest + '.py')
        if not path.exists():
            path.write_bytes((ROOT / 'src' / name).read_bytes())
    return {'train_sha256': frame_hash(train), 'test_sha256': frame_hash(test),
            'source_sha256': source,
            'git_commit_at_run': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {p: version(p) for p in ('numpy','pandas','scikit-learn','scipy','catboost','lightgbm','xgboost','optuna')}}


def save_result(result, path, config, provenance):
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'oof.npy', result.oof)
    np.save(path / 'test.npy', result.test)
    if result.raw_oof is not None:
        np.save(path / 'raw_oof.npy', result.raw_oof)
    arrays = {}
    for i, ((tr, va), pred) in enumerate(zip(result.split_indices, result.fold_predictions)):
        arrays.update({f'train_{i}': tr, f'valid_{i}': va, f'pred_{i}': pred})
    np.savez_compressed(path / 'folds.npz', **arrays)
    record = {**result.summary(), 'config': config, 'provenance': provenance,
              'artifacts_sha256': {p.name: sha256(p) for p in sorted(path.glob('*.np*'))}}
    (path / 'result.json').write_text(json.dumps(record, indent=2, default=str)+'\n')
    return record
