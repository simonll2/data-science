"""
train.py – Entraînement et évaluation des modèles.
"""

import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

from src.features import get_cat_indices

SEED = 42


def load_catboost_params(config_path: str = "configs/catboost.yaml") -> dict:
    """Charge les hyper-paramètres CatBoost depuis un fichier YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def cross_validate_model(model, X, y, n_splits: int = 5, cat_indices=None):
    """Évalue un modèle par KFold CV et renvoie les scores RMSE par fold.

    Les prédictions sont en log-space ; le RMSE est calculé après expm1.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rmse_scores = []
    oof_preds = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if isinstance(model, CatBoostRegressor) and cat_indices:
            model.fit(X_tr, y_tr, cat_features=cat_indices, verbose=0)
        else:
            model.fit(X_tr, y_tr)

        preds_log = model.predict(X_val)
        preds = np.expm1(preds_log)
        actual = np.expm1(y_val)

        rmse = np.sqrt(mean_squared_error(actual, preds))
        rmse_scores.append(rmse)
        oof_preds[val_idx] = preds_log

        print(f"  Fold {fold + 1} – RMSE : {rmse:.2f}")

    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    print(f"  => Moyenne RMSE : {mean_rmse:.2f} (+/- {std_rmse:.2f})\n")
    return rmse_scores, oof_preds


def get_baselines():
    """Renvoie les modèles baseline."""
    return {
        "DummyRegressor (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0, random_state=SEED),
    }


def get_catboost_model(config_path: str = "configs/catboost.yaml") -> CatBoostRegressor:
    """Instancie un CatBoostRegressor avec les paramètres du fichier YAML."""
    params = load_catboost_params(config_path)
    return CatBoostRegressor(**params, random_state=SEED, verbose=0)
