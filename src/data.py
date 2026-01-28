"""
data.py – Chargement des données Black Friday.
"""

import pandas as pd
from pathlib import Path


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge train.csv et test.csv depuis *data_dir* et renvoie (train, test)."""
    data_path = Path(data_dir)
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")
    return train, test


def quick_overview(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Affiche un résumé rapide du DataFrame."""
    print(f"=== {name} ===")
    print(f"Shape : {df.shape}")
    print(f"Colonnes : {list(df.columns)}")
    print(f"Types :\n{df.dtypes}\n")
    print(f"Valeurs manquantes :\n{df.isnull().sum()}\n")
    print(df.head())
