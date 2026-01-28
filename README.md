# Black Friday – Prédiction du montant d'achat

## Description

Projet de **Data Science / Machine Learning** réalisé dans le cadre de la dernière année du cycle ingénieur à l'**EFREI Paris**.

L'objectif est de **prédire le montant d'achat (`Purchase`)** de clients lors du Black Friday à partir de données démographiques et de catégories de produits. Le modèle principal utilisé est **CatBoost**, comparé à des baselines (DummyRegressor, Ridge).

## Arborescence

```
data-science/
├── notebook/
│   ├── black_friday.ipynb   # Notebook principal (EDA + modélisation)
│   ├── black_friday.html    # Export HTML
│   └── black_friday.pdf     # Export PDF
├── src/
│   ├── data.py              # Chargement des données
│   ├── preprocess.py        # Nettoyage et pré-traitement
│   ├── features.py          # Feature engineering
│   ├── train.py             # Entraînement et cross-validation
│   └── predict.py           # Prédiction et export submission
├── configs/
│   └── catboost.yaml        # Hyper-paramètres CatBoost
├── data/
│   ├── train.csv
│   └── test.csv
├── requirements.txt
├── README.md
└── submission.csv
```

## Instructions – Google Colab

1. **Cloner le dépôt** dans Colab :
   ```python
   !git clone https://github.com/<votre-user>/data-science.git
   %cd data-science
   ```

2. **Installer les dépendances** :
   ```python
   !pip install -r requirements.txt
   ```

3. **Charger les données** dans `data/` (upload `train.csv` et `test.csv`).

4. **Ouvrir et exécuter** `notebook/black_friday.ipynb` cellule par cellule.

## Exécution locale

```bash
pip install -r requirements.txt
jupyter notebook notebook/black_friday.ipynb
```

## Génération HTML / PDF

```bash
jupyter nbconvert --to html notebook/black_friday.ipynb
jupyter nbconvert --to pdf notebook/black_friday.ipynb
```

## Méthodologie

- **Transformation cible** : `log1p(Purchase)` pour stabiliser la variance
- **Validation** : KFold 5 folds, métrique RMSE
- **Feature engineering** : indicateurs de valeurs manquantes, gestion native des catégorielles par CatBoost
- **Interprétabilité** : SHAP values
- **Clustering** : KMeans + PCA sur agrégats utilisateurs

## Auteur

Projet académique – EFREI Paris, dernière année cycle ingénieur.
