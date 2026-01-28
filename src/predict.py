"""
predict.py – Prédiction sur le jeu de test et export submission.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from src.features import get_cat_indices


def predict_test(
    model: CatBoostRegressor,
    X_train,
    y_train,
    X_test,
    test_raw: pd.DataFrame,
    cat_indices: list[int],
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """Entraîne le modèle final sur tout le train et prédit sur le test.

    Les prédictions sont inversées de log-space via expm1.
    Génère submission.csv avec User_ID, Product_ID, Purchase.
    """
    model.fit(X_train, y_train, cat_features=cat_indices, verbose=0)

    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    submission = pd.DataFrame({
        "User_ID": test_raw["User_ID"],
        "Product_ID": test_raw["Product_ID"],
        "Purchase": preds,
    })
    submission.to_csv(output_path, index=False)
    print(f"Submission sauvegardée : {output_path} ({len(submission)} lignes)")
    return submission
