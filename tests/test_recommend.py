import numpy as np

from src.recommend import _same_folds


def _save_run(path, val_idx, preds):
    path.mkdir(parents=True)
    train_idx = np.setdiff1d(np.arange(10), val_idx)
    np.savez(path / "folds.npz", train_0=train_idx, val_0=val_idx, pred_0=preds)


def test_same_folds_compares_indices_not_predictions(tmp_path):
    _save_run(tmp_path / "model_a", np.array([8, 9]), np.array([1.0, 2.0]))
    _save_run(tmp_path / "model_b", np.array([8, 9]), np.array([5.0, 6.0]))  # same folds, other model
    _save_run(tmp_path / "other_split", np.array([7, 9]), np.array([1.0, 2.0]))  # same predictions, other folds
    assert _same_folds(tmp_path / "model_a", tmp_path / "model_b")
    assert not _same_folds(tmp_path / "model_a", tmp_path / "other_split")
