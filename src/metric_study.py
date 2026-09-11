"""Test scoring-aligned early stopping and the decoded price with boosters."""
from src.continue_study import evaluate
from src.data import load_raw
from src.config import TARGET


def main():
    tr,te=load_raw()
    for name in ['lattice_catboost','lattice_lightgbm','lattice_xgboost']:
        for weighted in [False,True]:
            evaluate(name,tr,te,f'metric/{name}_weighted_{weighted}',params={'weighted_training':weighted},repeats=3)
    evaluate('lattice_catboost',tr,te,'metric/lattice_catboost_units_stopping',params={'sales_early_stopping':False},repeats=3)

if __name__=='__main__':main()
