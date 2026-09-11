import numpy as np
import pandas as pd
import pytest

from src import train as train_script
from src.blend import fit_weights
from src.config import TARGET
from src.cv import prepare, rmse, run_cv
from src.data import Cleaner, clean
from src.features import FeatureBuilder, build_features, item_popularity
from src.models import SPECS_BY_NAME, library_status
from src.targets import TRANSFORMS
from tests.synthetic import make_synthetic


@pytest.fixture(scope="module")
def data():
    return make_synthetic(n_items=200, seed=0)


@pytest.fixture(scope="module")
def prepared(data):
    train, test, _ = data
    p = prepare(train, test)
    return p.train, p.test, p.matrices


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


def test_cleaner_fit_on_train_only_never_reads_test(data):
    train, test, _ = data
    cleaner = Cleaner().fit([train])
    # Weights come only from train rows: a product weighed only in test stays at the type median.
    assert set(cleaner.item_weight_.index) <= set(train["Item_Identifier"])
    out = cleaner.transform(test)
    assert out["Item_Weight"].notna().all() and (out["Item_Visibility"] > 0).all()
    assert out["Item_Weight_Missing"].sum() == test["Item_Weight"].isna().sum()
    assert out["Visibility_Was_Zero"].sum() == test["Item_Visibility"].eq(0).sum()


def test_feature_builder_handles_unseen_products(data):
    train, test, _ = data
    cleaner = Cleaner().fit([train])
    tr, te = cleaner.transform(train), cleaner.transform(test)
    builder = FeatureBuilder().fit([tr])
    te = te.assign(Item_Identifier="ZZZ99")  # every product unseen
    out = builder.transform(te)
    assert (out["Visibility_vs_Item_Mean"] == 1).all() and (out["Item_Store_Count"] == 1).all()
    assert np.isfinite(out["MRP_vs_Type_Median"]).all()


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


def test_catboost_runs_with_a_categorical_column_dropped(data):
    spec = SPECS_BY_NAME["catboost_raw"]
    if problem := library_status(spec.library):
        pytest.skip(problem)
    train, test, _ = data
    result = run_cv(spec, train, test, params={"iterations": 100}, n_splits=2, n_repeats=1,
                    drop={"Outlet_Identifier"})
    assert np.isfinite(result.oof_rmse)


def test_dropping_a_feature_group_removes_its_columns(data):
    from src.features import FEATURE_GROUPS

    train, test, _ = data
    drop = set(FEATURE_GROUPS["visibility"]) | set(FEATURE_GROUPS["outlet_id"])
    p = prepare(train, test, drop=drop)
    for name, (X_tr, _) in p.matrices.items():
        assert not any(c in X_tr.columns for c in ("Item_Visibility", "Visibility_vs_Item_Mean")), name
        assert not any(c.startswith("Outlet_Identifier") or c.startswith("Price_x_") for c in X_tr.columns), name


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


QUICK = {"lightgbm": {"n_estimators": 300}, "xgboost": {"n_estimators": 300},
         "catboost": {"iterations": 300}}


@pytest.mark.parametrize("name", sorted(SPECS_BY_NAME))
def test_every_model_beats_the_mean(data, name):
    spec = SPECS_BY_NAME[name]
    if problem := library_status(spec.library):
        pytest.skip(problem)
    train, test, _ = data
    result = run_cv(spec, train, test, params=QUICK.get(spec.library), n_splits=3, n_repeats=1)
    y = train[TARGET].to_numpy()
    assert result.oof_rmse < 0.8 * rmse(y, np.full_like(y, y.mean()))
    assert result.test.shape == (len(test),) and (result.test >= 0).all()
    if spec.iterations_param:
        assert len(result.best_iterations) == 3 and all(1 <= b <= 300 for b in result.best_iterations)


@pytest.mark.parametrize("scheme", ["holdout", "grouped"])
def test_other_validation_schemes(data, scheme):
    train, test, _ = data
    spec = SPECS_BY_NAME["ridge_interactions"]
    result = run_cv(spec, train, test, n_splits=3, n_repeats=1, scheme=scheme)
    y = train[TARGET].to_numpy()
    covered = ~np.isnan(result.oof)
    expected = len(train) if scheme == "grouped" else 0.2 * len(train)
    assert abs(covered.sum() - expected) <= 3  # stratified splits round per store
    assert result.oof_rmse < 0.85 * rmse(y, np.full_like(y, y.mean()))
    if scheme == "grouped":
        # Every product's rows fall in the same fold, so folds never share a product.
        items = train["Item_Identifier"]
        for tr_idx, va_idx in __import__("src.cv", fromlist=["folds"]).folds("grouped", train, 3, 1, 42):
            assert not set(items.iloc[tr_idx]) & set(items.iloc[va_idx])


def test_fold_fit_preprocessing_matches_prebuilt_closely(data):
    train, test, _ = data
    spec = SPECS_BY_NAME["ridge_interactions"]
    a = run_cv(spec, train, test, n_splits=3, n_repeats=1)
    b = run_cv(spec, train, test, n_splits=3, n_repeats=1, fold_fit=True)
    assert abs(a.oof_rmse - b.oof_rmse) < 0.05 * a.oof_rmse


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


def test_check_submission_accepts_good_files_and_names_problems(data, tmp_path):
    from src.submission import check_submission

    train, test, _ = data
    max_sales = float(train[TARGET].max())
    good = train_script.write_submission(test, np.full(len(test), 1000.0), tmp_path / "good.csv")
    assert check_submission(good, test, max_sales) == []

    bad = pd.read_csv(good).iloc[:-1]
    bad.loc[0, TARGET] = np.nan
    bad.to_csv(tmp_path / "bad.csv", index=True)  # index column, a missing row, a blank prediction
    problems = " | ".join(check_submission(tmp_path / "bad.csv", test, max_sales))
    assert "columns" in problems and "rows" in problems and "missing" in problems

    shuffled = pd.read_csv(good).sample(frac=1, random_state=0)
    shuffled.to_csv(tmp_path / "shuffled.csv", index=False)
    assert any("identifiers" in p for p in check_submission(tmp_path / "shuffled.csv", test, max_sales))


def test_refit_predicts_every_test_row(data, tmp_path):
    from src.refit import refit_predict

    train, test, _ = data
    pred = refit_predict("store_rate_popularity", train, test, outputs=tmp_path)
    assert pred.shape == (len(test),) and np.isfinite(pred).all() and (pred >= 0).all()
    if not library_status("catboost"):
        boosted = refit_predict("catboost_raw", train, test, seeds=2, rounds=50, outputs=tmp_path)
        assert boosted.shape == (len(test),) and np.isfinite(boosted).all()
    with pytest.raises(ValueError, match="rounds"):
        refit_predict("lightgbm_raw", train, test, outputs=tmp_path)


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
