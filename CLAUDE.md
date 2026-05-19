# CLAUDE.md — Concrete Compressive Strength (I2ML Project)

## Contexte projet

Projet académique pour le cours **I2ML** (Introduction to Machine Learning).

- **Dataset** : Concrete Compressive Strength (UCI) — régression supervisée
  - Fichier : `data/Concrete_Data.xls`
  - Cible : `Concrete compressive strength` (MPa)
  - 8 features continues (ciment, eau, agrégats, âge, etc.), ~1030 observations
- **Membres** : Tim & Arnaud
- **Livrable** : Jupyter Notebook (`notebooks/`) + Model Card (framework Mitchell et al.)

## Contraintes techniques non négociables

1. **sklearn Pipeline obligatoire** — tout preprocessing (scaling, imputation, etc.) doit être encapsulé dans un `sklearn.pipeline.Pipeline`. Aucun preprocessing à la main hors pipeline.
2. **Nested Cross-Validation** :
   - Inner loop : tuning des hyperparamètres (GridSearchCV ou RandomizedSearchCV)
   - Outer loop : estimation non biaisée de la Generalization Error (GE)
   - Ne jamais utiliser le même fold pour tuning ET évaluation finale
3. **Zéro data leakage** : aucun paramètre (moyenne, écart-type, etc.) calculé sur l'ensemble des données avant les boucles CV. Tout doit être `fit` uniquement sur les données d'entraînement de chaque fold.
4. **3 algorithmes de régression distincts** (exemples appropriés : Ridge, Random Forest Regressor, Gradient Boosting / XGBoost)
5. **Temps de réexécution raisonnable** : le notebook doit pouvoir tourner entièrement en quelques minutes à quelques heures, pas des jours. Limiter les grilles d'hyperparamètres en conséquence.

## Structure du repo

```
Concrete_Compressive_Strength/
├── CLAUDE.md                   # Ce fichier
├── README.md
├── data/
│   ├── Concrete_Data.xls       # Dataset principal
│   └── Concrete_Readme.txt     # Description des colonnes
├── docs/
│   └── readme                  # Documentation projet
├── notebooks/                  # Jupyter Notebooks (livrable principal)
├── syntheses/
│   └── arnaud/                 # Résumés de cours (Markdown)
│       ├── 1_ML_Basics.md
│       ├── 2_Regression_Supervise.md
│       ├── 3_Classification_Supervise.md
│       ├── 4 _Evaluation_de_performance.md
│       ├── 5_k-NN.md
│       ├── 6_Arbre_Classification_et_Regression.md
│       ├── 7_Random_Forest.md
│       ├── 8_Neural_Networks.md
│       ├── 9_Tuning.md
│       └── 10_Nested_Resampling.md   # CRITIQUE pour ce projet
└── .python-version
```

## Sources à consulter en priorité

1. **`syntheses/arnaud/10_Nested_Resampling.md`** — référence principale sur la nested CV
2. **`syntheses/arnaud/9_Tuning.md`** — hyperparameter tuning (GridSearch, RandomSearch)
3. **`syntheses/arnaud/2_Regression_Supervise.md`** — bases de la régression supervisée
4. **`syntheses/arnaud/7_Random_Forest.md`** — Random Forest pour la régression
5. **`syntheses/arnaud/4 _Evaluation_de_performance.md`** — métriques d'évaluation
6. **Cours I2ML** (GitHub public) : `https://github.com/slds-lmu/lecture_i2ml`
   - Exercices nested resampling : `exercises/nested-resampling/`
   - Exercices évaluation : `exercises/evaluation/`
   - Exercices régression : `exercises/supervised-regression/`

## Critères d'évaluation (pondération)

| Critère | Poids | Ce qui est attendu |
|---|---|---|
| Technical Rigor | 40% | Nested CV correct, Pipeline sklearn, zéro leakage |
| Methodological Depth | 30% | Justification hyperparamètres, choix learners, métriques |
| Specialized Focus | 15% | Pertinence par rapport au dataset béton (résistance compression) |
| Reporting Quality | 15% | Model Card claire, honnête, format Mitchell et al. |

## Comportement attendu de Claude

- Toujours vérifier l'absence de data leakage avant de valider du code
- Toujours encapsuler le preprocessing dans un Pipeline sklearn
- Justifier les choix d'hyperparamètres et de learners en lien avec les synthèses de cours
- Privilégier la correction méthodologique sur la performance brute des modèles
- Garder les grilles de recherche compactes pour des temps d'exécution raisonnables
- Référencer les synthèses de cours (`syntheses/arnaud/`) quand elles sont pertinentes
- La Model Card doit suivre le framework Mitchell et al. (2019)

## Rappel workflow nested CV (sklearn)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

# Pipeline = preprocessing + modèle
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SomeRegressor())
])

# Inner loop : tuning
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='neg_mean_squared_error')

# Outer loop : estimation GE non biaisée
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(search, X, y, cv=outer_cv, scoring='neg_mean_squared_error')
```

**Important** : `param_grid` pour un Pipeline préfixe les paramètres avec le nom de l'étape :
```python
param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [3, 5]}
```
