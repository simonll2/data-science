"""
features.py – Feature engineering pour le projet Black Friday.
"""

import pandas as pd

# Colonnes à exclure de la modélisation
ID_COLS = ["User_ID", "Product_ID"]

# Variables catégorielles gérées nativement par CatBoost
CAT_FEATURES = ["Gender", "Age", "City_Category"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construit les features de modélisation (sans la cible).

    - Supprime les identifiants (User_ID, Product_ID)
    - Encode Gender en binaire (F=0, M=1)
    - Garde les variables catégorielles en string pour CatBoost
    """
    df = df.copy()

    # Suppression des identifiants
    df = df.drop(columns=[c for c in ID_COLS if c in df.columns])

    # Encodage binaire du genre
    df["Gender"] = df["Gender"].map({"F": 0, "M": 1}).astype(int)

    # S'assurer que les catégorielles sont bien en string pour CatBoost
    df["Age"] = df["Age"].astype(str)
    df["City_Category"] = df["City_Category"].astype(str)

    return df


def get_cat_indices(df: pd.DataFrame) -> list[int]:
    """Renvoie les indices des colonnes catégorielles dans le DataFrame."""
    cat_cols = ["Age", "City_Category"]
    return [df.columns.get_loc(c) for c in cat_cols if c in df.columns]
