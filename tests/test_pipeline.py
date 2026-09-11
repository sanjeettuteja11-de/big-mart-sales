import numpy as np
import pandas as pd
import pytest

from src import train as train_script
from src.blend import fit_weights
from src.config import TARGET
from src.cv import build_matrices, rmse, run_cv
from src.data import clean
from src.features import build_features, item_popularity
from src.models import SPECS_BY_NAME, library_status
from src.targets import TRANSFORMS
from tests.synthetic import make_synthetic


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n_items=200, seed=0)


@pytest.fixture(scope="module")
def prepared(data):
    train, test, _ = data
    tr, te = build_features(*clean(train, test))
    return tr, te, build_matrices(tr, te)


def test_clean_fixes_every_quirk(data):
    train, test, _ = data
    tr, te = clean(train, test)
    both = pd.concat([tr, te])
    assert both["Item_Weight"].notna().all()
    assert both["Outlet_Size"].notna().all()
    assert (both["Item_Visibility"] > 0).all()
    assert set(both["Item_Fat_Content"]) <= {"Low Fat", "Regular", "Non-Edible"}
    assert (len(tr), len(te)) == (len(train), len(test))
    assert TARGET not in te.columns


def test_missing_weight_borrowed_from_same_product(data):
    both = pd.concat(clean(*data[:2]))
    assert both.groupby("Item_Identifier")["Item_Weight"].nunique().max() == 1


def test_features_do_not_depend_on_sales(data):
    train, test, _ = data
    shuffled = train.assign(
        **{TARGET: np.random.default_rng(1).permutation(train[TARGET].to_numpy())}
    )
    a, _ = build_features(*clean(train, test))
    b, _ = build_features(*clean(shuffled, test))
    pd.testing.assert_frame_equal(a.drop(columns=TARGET), b.drop(columns=TARGET))


def test_matrices_align_and_are_finite(prepared):
    tr, te, matrices = prepared
    for name, (X_tr, X_te) in matrices.items():
        assert list(X_tr.columns) == list(X_te.columns), name
        assert (len(X_tr), len(X_te)) == (len(tr), len(te)), name
        assert np.isfinite(X_tr.select_dtypes("number").to_numpy()).all(), name


@pytest.mark.parametrize("name", sorted(TRANSFORMS))
def test_target_transforms_round_trip(prepared, name):
    tr, _, _ = prepared
    y, t = tr[TARGET].to_numpy(), TRANSFORMS[name]
    np.testing.assert_allclose(t.inverse(t.forward(y, tr), tr), y, rtol=1e-9)


def test_weighted_units_loss_equals_raw_sales_loss(prepared):
    tr, _, _ = prepared
    y, t, mrp = tr[TARGET].to_numpy(), TRANSFORMS["units"], tr["Item_MRP"].to_numpy()
    pred = 0.9 * y + 50
    raw_loss = np.mean((y - pred) ** 2)
    units_loss = np.mean(t.weight(tr) * (t.forward(y, tr) - pred / mrp) ** 2) * np.mean(mrp**2)
    assert units_loss == pytest.approx(raw_loss)


def test_item_popularity_never_sees_its_own_row(prepared):
    tr, te, _ = prepared
    sub = tr.iloc[:150]
    only_once = sub["Item_Identifier"].map(sub["Item_Identifier"].value_counts()).eq(1).to_numpy()
    # Leave-one-out: a product with a single row has nothing else to learn from.
    values, (test_values,) = item_popularity(sub, sub[TARGET].to_numpy(), [te], n_splits=len(sub))
    assert only_once.any() and np.all(values[only_once] == 1.0)
    assert np.isfinite(test_values).all()


QUICK = {"lightgbm": {"n_estimators": 150}, "xgboost": {"n_estimators": 150},
         "catboost": {"iterations": 200}}


@pytest.mark.parametrize("name", sorted(SPECS_BY_NAME))
def test_every_model_beats_the_mean(prepared, name):
    spec = SPECS_BY_NAME[name]
    if problem := library_status(spec.library):
        pytest.skip(problem)
    tr, te, matrices = prepared
    result = run_cv(spec, tr, te, matrices, QUICK.get(spec.library), n_splits=3, n_repeats=1)
    y = tr[TARGET].to_numpy()
    assert result.oof_rmse < 0.8 * rmse(y, np.full_like(y, y.mean()))
    assert result.test.shape == (len(te),) and (result.test >= 0).all()


def test_blend_weights_favour_the_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    P = np.column_stack([y + rng.normal(scale=s, size=1000) for s in (0.5, 1.0, 2.0)])
    w = fit_weights(P, y)
    assert w.min() >= 0 and w.sum() == pytest.approx(1)
    assert w[0] > w[1] > w[2]


def test_end_to_end_writes_a_valid_submission(data, tmp_path):
    train, test, truth = data
    report = train_script.run(
        train, test, ["ridge_interactions", "hist_gb_units"], n_splits=3, n_repeats=1,
        outputs=tmp_path / "outputs", submissions=tmp_path / "submissions",
    )
    sub = pd.read_csv(tmp_path / "submissions" / report["submissions"]["recommended"])
    assert list(sub.columns) == ["Item_Identifier", "Outlet_Identifier", "Item_Outlet_Sales"]
    assert len(sub) == len(test) and (sub["Item_Outlet_Sales"] >= 0).all()
    baseline = rmse(truth, np.full_like(truth, train[TARGET].mean()))
    assert rmse(truth, sub["Item_Outlet_Sales"]) < 0.8 * baseline


def test_recover_units_rebuilds_every_sale():
    from src.structure import recover_units

    rng = np.random.default_rng(0)
    mrp = rng.uniform(31, 267, 5000).round(4)
    units = rng.integers(1, 60, 5000)
    offset = rng.integers(-20, 21, 5000) / 10
    sales = (units * (mrp + offset)).round(4)
    got_units, got_offset = recover_units(sales, mrp)
    # A sale can occasionally split two ways (30 x 62 = 31 x 60); every split
    # found must still rebuild the sale, and nearly all must be the true one.
    np.testing.assert_allclose(got_units * (mrp + got_offset), sales, atol=0.01)
    assert np.mean(got_units == units) > 0.97


def test_store_rate_popularity_learns_rates_and_shrinks():
    from src.structure import StoreRatePopularity

    X = pd.DataFrame({"Outlet_Identifier": ["A"] * 4 + ["B"] * 4,
                      "Item_Identifier": ["x", "y", "x", "y", "x", "y", "x", "y"]})
    y = np.array([12, 8, 12, 8, 3, 2, 3, 2], dtype=float)  # store A 10/row, B 2.5/row; x sells 1.2x
    unshrunk = StoreRatePopularity(smoothing=0).fit(X, y)
    np.testing.assert_allclose(unshrunk.predict(X), y)
    shrunk = StoreRatePopularity(smoothing=1000).fit(X, y).predict(X)
    np.testing.assert_allclose(shrunk, [10, 10, 10, 10, 2.5, 2.5, 2.5, 2.5], rtol=0.01)


def test_store_rate_variants_pool_by_type_and_drop_the_product_factor():
    from src.structure import StoreRatePopularity

    X = pd.DataFrame({"Outlet_Identifier": ["A", "A", "B", "B"],
                      "Outlet_Type": ["T", "T", "T", "T"],
                      "Item_Identifier": ["x", "y", "x", "y"]})
    y = np.array([12, 8, 6, 4], dtype=float)
    plain = StoreRatePopularity(smoothing=float("inf")).fit(X, y)
    np.testing.assert_allclose(plain.predict(X), [10, 10, 5, 5])
    pooled = StoreRatePopularity(smoothing=float("inf"), rate_level="type").fit(X, y)
    np.testing.assert_allclose(pooled.predict(X), [7.5] * 4)
