"""
preprocess.py – Nettoyage et pré-traitement des données Black Friday.
"""

import pandas as pd
import numpy as np


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage commun appliqué à train et test.

    - Remplacement des NA de Product_Category_2/3 par "Unknown"
    - Ajout d'indicateurs de valeurs manquantes (PC2_missing, PC3_missing)
    - Conversion de Stay_In_Current_City_Years en numérique
    """
    df = df.copy()

    # Indicateurs de valeurs manquantes
    df["PC2_missing"] = df["Product_Category_2"].isnull().astype(int)
    df["PC3_missing"] = df["Product_Category_3"].isnull().astype(int)

    # Remplacement des NA
    df["Product_Category_2"] = df["Product_Category_2"].fillna(-1).astype(int)
    df["Product_Category_3"] = df["Product_Category_3"].fillna(-1).astype(int)

    # Stay_In_Current_City_Years : '4+' -> 4, conversion en int
    df["Stay_In_Current_City_Years"] = (
        df["Stay_In_Current_City_Years"]
        .astype(str)
        .str.replace("+", "", regex=False)
        .astype(int)
    )

    return df


def split_features_target(
    df: pd.DataFrame,
    target: str = "Purchase",
    log_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Sépare features et cible. Applique log1p si demandé."""
    if target in df.columns:
        y = df[target].copy()
        if log_target:
            y = np.log1p(y)
        X = df.drop(columns=[target])
    else:
        y = None
        X = df.copy()
    return X, y
