# Historique des versions — Concrete Compressive Strength

*Projet I2ML — Arnaud & Tim — HELMo Bloc 2, mai 2026*

---

## V1 — Versions individuelles

### V1.A — Arnaud

| Champ | Détail |
|---|---|
| **Fichiers** | `notebooks/arnaud/01_eda_data_cleaning.ipynb`, `notebooks/arnaud/02_nested_cv_models.ipynb` |
| **Modèles** | Ridge, Random Forest, Gradient Boosting |
| **Inner CV** | 5-fold, `random_state=42` |
| **Outer CV** | 5-fold, `random_state=0` |
| **Grilles** | Compactes (Ridge ×5, RF ×12, GB ×8) |

#### Résultats (nested CV, 5-fold outer)

| Modèle | RMSE moyen | Std | R² moyen |
|---|---|---|---|
| Ridge | 10.385 MPa | ± 0.449 | 0.590 |
| Random Forest | 4.989 MPa | ± 0.314 | 0.905 |
| **Gradient Boosting** | **4.584 MPa** | **± 0.380** | **0.919** |

---

### V1.T — Tim

| Champ | Détail |
|---|---|
| **Fichiers** | `notebooks/tim/concrete_strength.ipynb`, `docs/model_card_tim.md` |
| **Modèles** | Ridge, Random Forest, Gradient Boosting, XGBoost (bonus) |
| **Inner CV** | 3-fold, `random_state=42` |
| **Outer CV** | 5-fold, `random_state=42` |
| **Grilles** | Larges (RF ×72, GB ×72, XGB ×72) |

#### Résultats (nested CV, 5-fold outer)

| Modèle | RMSE moyen | Std | R² moyen |
|---|---|---|---|
| Ridge | 10.505 MPa | ± 0.809 | — |
| Random Forest | 5.077 MPa | ± 0.498 | — |
| **Gradient Boosting** | **4.200 MPa** | **± 0.285** | — |
| XGBoost | 4.253 MPa | ± 0.328 | — |

---

## V2 — Version finale (merge)

> Deux exports de la V2 existent :
> - **`notebooks/final/`** — version complète avec toutes les explications (défense orale, synthèses, justifications)
> - **`notebooks/prof/`** — version épurée pour le livrable prof (sections + code + résultats, sans la documentation orale)


| Champ | Détail |
|---|---|
| **Fichiers** | `notebooks/final/01_eda_data_cleaning.ipynb`, `notebooks/final/02_nested_cv_models.ipynb` |
| **Annexe** | `notebooks/final/03_bonus_xgboost.ipynb` |
| **Modèles** | Ridge, Random Forest, Gradient Boosting |
| **Inner CV** | 5-fold, `random_state=42` |
| **Outer CV** | 5-fold, `random_state=0` |

#### Résultats (nested CV, 5-fold outer)

| Modèle | RMSE moyen | Std | R² moyen |
|---|---|---|---|
| Ridge | 10.385 MPa | ± 0.449 | 0.590 |
| Random Forest | 4.935 MPa | ± 0.318 | 0.907 |
| **Gradient Boosting** | **4.208 MPa** | **± 0.261** | **0.932** |

#### Meilleurs hyperparamètres (fit sur tout le dataset)

| Modèle | Hyperparamètres |
|---|---|
| Ridge | `alpha=1` |
| Random Forest | `max_depth=20`, `min_samples_leaf=1`, `min_samples_split=2`, `n_estimators=300` |
| **Gradient Boosting** | `learning_rate=0.2`, `max_depth=4`, `n_estimators=300`, `subsample=1.0` |
