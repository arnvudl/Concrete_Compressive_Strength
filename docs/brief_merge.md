# Brief — Comparaison des approches & stratégie de merge

*Arnaud × Tim — Concrete Compressive Strength (I2ML, mai 2026)*

---

## 1. Ce qu'on a chacun construit

### Arnaud

| Ce qui est fait | Détail |
|---|---|
| **Structure en 2 notebooks** | `01_eda_data_cleaning.ipynb` + `02_nested_cv_models.ipynb` |
| **EDA + nettoyage complet** | Doublons supprimés, boxplots, heatmap, QQ-plot, export `concrete_clean.csv` |
| **Documentation orale** | Chaque section a une justification théorique (références aux synthèses) pensée pour la défense |
| **Théorie CART (§3.5)** | Expliqué entre Ridge et RF — base commune de RF et GB |
| **Specialized Focus (§3.3)** | Pourquoi l'interprétabilité est critique en génie civil, tableau learner exclus / inclus |
| **Model Card** | Format Mitchell et al. 2019, pour le meilleur modèle uniquement (dans le notebook) |
| **Seeds différenciés** | `inner_cv` → `random_state=42`, `outer_cv` → `random_state=0` (logique !) |
| **Résultats** | Notebook exécuté — Ridge : 10.385 MPa · RF : 4.989 MPa · **GB : 4.584 MPa** |

### Tim

| Ce qui est fait | Détail |
|---|---|
| **Résultats réels** | Ridge : 10.5 ± 0.8 MPa · RF : 5.1 ± 0.5 MPa · **GB : 4.2 ± 0.3 MPa** · XGBoost : 4.3 ± 0.3 MPa |
| **Grilles d'hyperparamètres plus larges** | RF : 72 combinaisons, GB : 72 combinaisons |
| **XGBoost (bonus)** | 4ᵉ modèle testé, résultats cohérents avec GB |
| **Section Final Model** | Modèle réentraîné sur tout le dataset avec les meilleurs HPs |
| **Model Card** | `docs/model_card_tim.md` — format Mitchell et al. 2019, GB comme meilleur modèle |

---

## 2. Ce qu'on peut combiner

L'idée du merge, c'est de garder le meilleur des deux mondes :

| Élément | Source retenue | Raison |
|---|---|---|
| Structure 2 notebooks | Arnaud | Plus lisible, EDA séparé |
| Grilles d'hyperparamètres | Tim (élargies) | Plus de configs = meilleure couverture |
| Résultats numériques | Tim | Déjà exécutés et vérifiés |
| XGBoost | Tim | Bonus apprécié, déjà codé |
| Section Final Model | Tim | Bonne pratique, attendu implicitement |
| Documentation / justifications | Arnaud | Nécessaire pour la défense orale |
| Specialized Focus | Arnaud | Manquant chez Tim (15% des points) |
| Model Card | Les deux ✅ | Arnaud (notebook §9) + Tim (`docs/model_card_tim.md`) — à fusionner |
| CART théorie | Arnaud | Transition Ridge → RF → GB plus fluide |

---

## 3. Une petite chose à harmoniser côté Tim

Dans `concrete_strength.ipynb`, les deux boucles CV utilisent le même seed :

```python
# Tel quel chez Tim
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
```

C'est pas un bug grave, mais en nested CV, utiliser des seeds **différents** pour les deux boucles est une bonne pratique : ça garantit que les partitions internes et externes sont vraiment indépendantes, et ça rend l'estimation de la GE un peu plus robuste. Dans le notebook fusionné, on peut passer `outer_cv` à `random_state=0` comme ça :

```python
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv  = KFold(n_splits=5, shuffle=True, random_state=0)
```

> Note : on en profite aussi pour passer les deux boucles à **5 folds** (Tim avait 3 folds en interne, 5 en externe). Avec 5×5 c'est plus symétrique et les résultats sont légèrement plus stables.

---

## 4. Proposition de structure du notebook fusionné

```
notebooks/
└── final/
    ├── 01_eda_data_cleaning.ipynb       # Base : Arnaud
    └── 02_nested_cv_models.ipynb        # Merge des deux
        ├── §0  Imports
        ├── §1  Chargement des données
        ├── §2  Framework nested CV (docs Arnaud)
        ├── §3  Ridge Regression
        ├── §3.5 CART — base théorique (Arnaud)
        ├── §4  Random Forest (grille Tim, docs Arnaud)
        ├── §5  Gradient Boosting (grille Tim, docs Arnaud)
        ├── §5.5 XGBoost (bonus Tim)
        ├── §6  Comparaison des modèles
        ├── §7  Final Model — réentraînement (Tim)
        ├── §8  Specialized Focus — interprétabilité (Arnaud)
        └── §9  Model Card — meilleur modèle (Arnaud)
```

---

## 5. Répartition pour la défense orale

| Partie | Qui défend | Pourquoi |
|---|---|---|
| EDA + nettoyage | Arnaud | Il l'a conçu |
| Framework nested CV (théorie) | Arnaud | Synthèses détaillées |
| Ridge + CART | Arnaud | Docs en place |
| Random Forest | Tim ou ensemble | Tim a les résultats, Arnaud la théorie |
| Gradient Boosting + XGBoost | Tim | Il a les résultats et XGBoost |
| Specialized Focus | Arnaud | Il l'a rédigé |
| Model Card | Arnaud | Il l'a rédigé |

---

---

## 6. Tableau comparatif des résultats

Les deux notebooks ont été exécutés. Les différences de résultats s'expliquent principalement par les grilles d'HPs et les seeds (voir tableau params ci-dessous).

### RMSE outer CV (MPa) — plus bas = meilleur

| Modèle | Tim — RMSE moyen | Tim — std | Arnaud — RMSE moyen | Arnaud — std |
|---|---|---|---|---|
| Ridge | 10.505 MPa | ± 0.809 | 10.385 MPa | ± 0.449 |
| Random Forest | 5.077 MPa | ± 0.498 | 4.989 MPa | ± 0.314 |
| **Gradient Boosting** | **4.200 MPa** | **± 0.285** | 4.584 MPa | ± 0.380 |
| XGBoost | 4.253 MPa | ± 0.328 | *(non testé)* | — |

> GB gagne chez les deux — cohérence forte. L'écart (~0.4 MPa) vient des grilles plus larges de Tim. R² Arnaud : Ridge=0.590, RF=0.905, GB=0.919.

### R² outer CV (pour info)

| Modèle | Arnaud — R² moyen | Arnaud — std |
|---|---|---|
| Ridge | 0.590 | ± 0.031 |
| Random Forest | 0.905 | ± 0.014 |
| Gradient Boosting | 0.919 | ± 0.015 |

### Paramètres des boucles CV

| Paramètre | Tim | Arnaud |
|---|---|---|
| Folds inner CV | 3 | 5 |
| Folds outer CV | 5 | 5 |
| `random_state` inner | 42 | 42 |
| `random_state` outer | 42 ⚠️ (identique) | 0 ✅ (différencié) |
| Taille grille RF | 72 combos | 12 combos |
| Taille grille GB | 72 combos | 8 combos |
| Scorer inner | `neg_mean_squared_error` | `neg_mean_squared_error` |
| Scorer outer (rapporté) | RMSE (√\|MSE\|) | RMSE (√\|MSE\|) |

### Configuration des grilles d'hyperparamètres

| Modèle | Hyperparamètre | Tim | Arnaud |
|---|---|---|---|
| **Ridge** | `alpha` | *non précisé* | [0.001, 0.1, 1, 10, 100] |
| **RF** | `n_estimators` | [100, 200, 300] | [50, 100] |
| **RF** | `max_depth` | [None, 10, 20, 30] | [5, 10, None] |
| **RF** | `min_samples_split` | [2, 5, 10] | — |
| **RF** | `min_samples_leaf` | [1, 2] | [1, 5] |
| **GB** | `n_estimators` | [100, 200, 300] | [50, 100] |
| **GB** | `learning_rate` | [0.01, 0.05, 0.1] | [0.05, 0.1] |
| **GB** | `max_depth` | [3, 5, 7] | [3, 5] |

> Pour le merge, on adopte les grilles de Tim (plus larges) avec les seeds d'Arnaud (différenciés).

---

*Ce brief est une base de discussion — à adapter ensemble selon vos préférences.*
