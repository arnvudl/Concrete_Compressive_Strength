# Fiches orales — Concrete Compressive Strength

> **Mode d'emploi :** chaque fiche est **auto-suffisante**. Tu n'as pas besoin de relire le cours pour comprendre. Tout est ici : la définition, le pourquoi, le comment, l'analogie, et le lien avec notre projet.

---

## Ordre de Lecture Conseillé

```mermaid
graph LR
    F00["00\nLe Projet\nDataset + béton"] --> F01["01\nErreur de\nGénéralisation"]
    F01 --> F02["02\nMétriques\nRMSE, R²"]
    F02 --> F03["03\nCross-\nValidation"]
    F03 --> F04["04\nHyper-\nparamètres"]
    F04 --> F05["05\nNested CV\n⭐ critique"]
    F05 --> F06["06\nPipeline\nanti-leakage"]
    F06 --> F07["07\nRidge\nRégression"]
    F07 --> F08["08\nCART, RF\net GB"]
    F08 --> F09["09\nModel Card\n9 sections"]
    F09 --> F10["10\nInterprét-\nabilité"]
```

---

## Index des Fiches

| Fiche | Sujet | Concepts clés | Points à retenir |
|---|---|---|---|
| [00](00_projet_dataset.md) | **Le projet & le dataset** | 8 features, physique du béton, résultats finaux | Loi de Féret, hydratation log(age), RMSE = 4.2 MPa |
| [01](01_erreur_generalisation.md) | **Erreur de généralisation** | Train error biaisé, GE, biais-variance | Train error ≠ GE, 1-NN = 0% train mais overfitting |
| [02](02_metriques_regression.md) | **Métriques de régression** | RMSE, R², MSE, neg_MSE | RMSE en MPa, neg_MSE dans GridSearchCV |
| [03](03_cross_validation.md) | **Cross-Validation** | 5-fold, shuffle, biais pessimiste | shuffle=True, seeds différents, 80% = pessimiste |
| [04](04_hyperparametres_tuning.md) | **Hyperparamètres & Tuning** | θ vs λ, GridSearch vs RandomSearch | Préfixe `model__`, grilles compactes |
| [05](05_nested_cv.md) | **Nested Cross-Validation** | Overtuning, loot box, 2 boucles | `cross_val_score(GridSearchCV(...))` |
| [06](06_pipeline_data_leakage.md) | **Pipeline & Data Leakage** | Comment le leakage arrive, Pipeline sklearn | `.fit()` sur train seulement, `model__` |
| [07](07_ridge_regression.md) | **Ridge Regression** | L2, StandardScaler obligatoire, coefficients | cement=+12.05, water=-3.37, alpha=1 |
| [08](08_cart_rf_gb.md) | **CART, Random Forest, GB** | Bagging vs Boosting, variance vs biais | RF réduit variance, GB réduit biais |
| [09](09_model_card.md) | **Model Card (Mitchell 2019)** | 9 sections, best model, limites | GB 4.2 MPa, extrapolation impossible |
| [10](10_interpretabilite.md) | **Interprétabilité** | Prédiction vs Explication, Feature Importance | age 35% + cement 29% = 64%, pourquoi pas NN |

---

## 10 Questions les Plus Probables à l'Oral

### 1. "Pourquoi la nested CV et pas une simple CV ?"
→ **Fiche 05**

La simple CV sélectionne le minimum parmi plusieurs évaluations bruitées → toujours trop optimiste. Analogie loot box : cherry-picker la meilleure sur 72 tirages. Nested CV = boucle externe jamais vue pendant le tuning.

---

### 2. "Comment votre Pipeline évite-t-il le data leakage ?"
→ **Fiche 06**

Sans Pipeline, `scaler.fit_transform(X)` calcule μ, σ sur tout X (train + test). Avec Pipeline, sklearn appelle `.fit()` seulement sur X_train de chaque fold. Garanti par construction.

---

### 3. "Pourquoi RMSE et pas MSE ?"
→ **Fiche 02**

RMSE est en MPa = même unité que la target. Un ingénieur comprend "erreur de 4.2 MPa". Il ne comprend pas "erreur de 17.7 MPa²". MSE utilisé en interne (GridSearchCV) car dérivable.

---

### 4. "Pourquoi Gradient Boosting est meilleur que Random Forest ?"
→ **Fiche 08**

RF réduit la variance (arbres parallèles indépendants). GB réduit le biais (arbres séquentiels qui corrigent les résidus). Les relations physiques du béton (log(age), loi de Féret) ont un biais résiduel que GB corrige itérativement. Résultat : 4.2 vs 4.9 MPa.

---

### 5. "Pourquoi avez-vous exclu les réseaux de neurones ?"
→ **Fiche 10**

Interprétabilité critique en génie civil (normes, bureau de contrôle). GB donne Feature Importances, Ridge donne des coefficients directs. Les NN sont boîtes noires. Gain potentiel marginal sur 1005 obs.

---

### 6. "Que signifie le biais pessimiste de votre RMSE ?"
→ **Fiches 01 + 03**

En 5-fold CV, chaque modèle intermédiaire est entraîné sur 80% des données. Plus de données → généralement meilleur. Le modèle final (100%) est légèrement meilleur. Notre RMSE = 4.2 MPa est légèrement au-dessus de la vraie performance finale.

---

### 7. "Expliquez les 9 sections de Mitchell et al."
→ **Fiche 09**

Détails, Usage prévu, Facteurs de variation, Métriques, Données test, Données train, Chiffres, Éthique, Limites. La Model Card force à être honnête sur les limites autant que les performances.

---

### 8. "Que vaut concrètement votre RMSE de 4.2 MPa ?"
→ **Fiches 00 + 02**

Sur un béton de 40 MPa → erreur typique ±4.2 MPa (10.5%). Acceptable pour aide à la formulation en laboratoire. NON acceptable pour décision structurelle (nécessite essais physiques).

---

### 9. "Pourquoi shuffle=True dans votre KFold ?"
→ **Fiche 03**

Le dataset UCI Concrete est ordonné par formulation. Sans shuffle, fold 1 = formulations 1-201 (une gamme), fold 2 = 202-402 (autre gamme). Les folds ne seraient pas représentatifs de la distribution globale.

---

### 10. "Qu'est-ce que l'overtuning et comment l'évitez-vous ?"
→ **Fiches 04 + 05**

L'overtuning = cherry-picker la meilleure config parmi N évaluations bruitées → minimum statistiquement biaisé. Preuve : classifieur aléatoire (GE=50%) avec 100 configs → score apparent 38%. Solution : nested CV (boucle externe jamais vue pendant le tuning).

---

## Rappel des Résultats Clés

| Modèle | RMSE outer CV | R² | HPs optimaux |
|---|---|---|---|
| Ridge | 10.385 ± 0.449 MPa | 0.590 | alpha=1 |
| Random Forest | 4.935 ± 0.318 MPa | 0.907 | max_depth=20, n_estimators=300 |
| **Gradient Boosting** | **4.208 ± 0.261 MPa** | **0.932** | lr=0.2, depth=4, n=300 |

**Top Feature Importances GB :** age (35%) → cement (29%) → water (11%) → slag (8.5%)
