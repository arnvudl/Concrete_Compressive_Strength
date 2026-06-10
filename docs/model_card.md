# Model Card : Concrete Compressive Strength Predictor

> Framework : Mitchell et al. (2019), *Model Cards for Model Reporting*  
> Référence : https://arxiv.org/abs/1810.03993  
> Projet : I2ML, HELMo Bloc 2, mai 2026  
> Auteurs : Arnaud & Tim

---

## 1. Model Details

| Champ | Valeur |
|---|---|
| **Algorithme** | Gradient Boosting Regressor (sklearn `GradientBoostingRegressor`) |
| **Tâche** | Régression supervisée, prédiction de la résistance à la compression (MPa) |
| **Version** | V2, version finale (merge Arnaud × Tim) |
| **Date** | Mai 2026 |
| **Développeurs** | Arnaud & Tim, HELMo Bloc 2, cours I2ML |
| **Framework** | scikit-learn, `Pipeline(StandardScaler → GradientBoostingRegressor)` |
| **Hyperparamètres retenus** | `learning_rate=0.2`, `max_depth=4`, `n_estimators=300`, `subsample=1.0` |
| **Protocole d'évaluation** | Nested Cross-Validation 5-fold outer × 5-fold inner (GridSearchCV) |
| **Repo** | https://github.com/arnvudl/Concrete_Compressive_Strength |

### Pourquoi Gradient Boosting comme meilleur modèle ?

On a sélectionné le modèle sur la base du **RMSE outer CV**, la seule métrique non biaisée du projet (les données de test n'ont à aucun moment participé au tuning).

| Modèle | RMSE moyen | Std | R² moyen | Rang |
|---|---|---|---|---|
| Ridge | 10.385 MPa | ± 0.449 | 0.590 | 3 |
| Random Forest | 4.935 MPa | ± 0.318 | 0.907 | 2 |
| **Gradient Boosting** | **4.208 MPa** | **± 0.261** | **0.932** | **1** |

GB fait mieux que RF grâce au boosting séquentiel : chaque arbre corrige les erreurs du précédent, ce qui réduit le biais étape par étape. Les relations entre composition et résistance sont fortement non-linéaires (loi eau/ciment, hydratation logarithmique de l'âge), et les modèles à base d'arbres capturent ça naturellement, là où Ridge échoue.

---

## 2. Intended Use

### Usages prévus

- **Aide à la formulation** : estimer la résistance d'une formulation de béton avant les essais physiques, pour réduire le nombre d'éprouvettes nécessaires.
- **Analyse de sensibilité** : voir l'impact de chaque composant (ciment, eau, âge) sur la résistance via les Feature Importances.
- **Recherche académique** : démonstration d'une pipeline ML complète (nested CV + interprétabilité) sur des données réelles de génie civil.

### Utilisateurs cibles

Étudiants, chercheurs et ingénieurs qui veulent un outil de prédiction rapide pour des formulations de béton standards.

### Usages hors périmètre

- ❌ Validation structurelle réglementaire (le modèle n'est pas certifié, des essais physiques restent obligatoires).
- ❌ Formulations hors de la plage du dataset (voir §9, les arbres ne peuvent pas extrapoler).
- ❌ Bétons spéciaux : béton fibré, béton projeté, béton ultra-haute performance, béton auto-plaçant avec adjuvants non représentés dans les données.
- ❌ Prise de décision seule sur des ouvrages critiques (ponts, bâtiments de grande hauteur) sans tests physiques en plus.

---

## 3. Factors

### Features d'entrée

| Feature | Unité | Min | Max | Moyenne | Notes |
|---|---|---|---|---|---|
| `cement` | kg/m³ | 102 | 540 | 281 | Liant principal |
| `slag` | kg/m³ | 0 | 359 | 74 | ~70% de valeurs = 0 |
| `fly_ash` | kg/m³ | 0 | 200 | 54 | ~55% de valeurs = 0 |
| `water` | kg/m³ | 122 | 247 | 182 | Ratio eau/ciment clé |
| `superplasticizer` | kg/m³ | 0 | 32 | 6 | ~40% de valeurs = 0 |
| `coarse_agg` | kg/m³ | 801 | 1145 | 972 | Granulats grossiers |
| `fine_agg` | kg/m³ | 594 | 993 | 774 | Granulats fins |
| `age` | jours | 1 | 365 | 46 | Fortement concentré à 28j |

### Facteurs de variation identifiés

- **`age`** (35% d'importance GB) et **`cement`** (29%) dominent. Ensemble, ils représentent environ 64% de l'importance totale du modèle.
- **`slag`, `fly_ash`, `superplasticizer`** : distributions bimodales avec beaucoup de zéros (adjuvants absents dans plein de formulations), ce qui peut dégrader la prédiction pour des formulations à forte concentration.
- **Formulations à `age` rare** : le dataset est très concentré autour de 28 jours (standard industriel). Pour des âges peu courants (1 jour ou 365 jours), la couverture est faible et les prédictions moins fiables.

### Facteurs non pris en compte

- Conditions de cure (température, humidité ambiante)
- Provenance des matériaux (variabilité du ciment selon le fournisseur)
- Technique de mise en place (vibration, compaction)

---

## 4. Metrics

### Métrique principale

**RMSE, exprimé en MPa.**

On a choisi cette métrique parce qu'elle donne l'erreur dans la même unité que la cible (MPa), ce qui la rend directement lisible pour un ingénieur civil. Les normes béton (EN 206, NF EN 197) raisonnent en classes de résistance en MPa.

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

### Métrique complémentaire

**R², coefficient de détermination.**

Mesure la fraction de variance de la résistance expliquée par le modèle. R²=1 = prédictions parfaites, R²=0 = équivalent à prédire la moyenne. Pas dépendant de l'unité.

### Métrique interne (inner loop)

**neg_MSE** utilisé dans GridSearchCV pour le tuning. Cohérent avec RMSE car √neg_MSE = RMSE.

### Seuil de décision

Pas de seuil fixé, c'est une tâche de régression continue. Pour donner un ordre d'idée : un RMSE < 5 MPa sur un dataset dont la cible varie de 2 à 82 MPa (range = 80 MPa) représente une erreur d'environ 6%, ce qui est acceptable pour de l'aide à la formulation.

### Note sur le biais pessimiste

Les RMSE reportés sont légèrement **pessimistes** : ils estiment la performance d'un modèle entraîné sur 80% des données (4 folds sur 5). Le modèle final, entraîné sur tout le dataset, serait un peu meilleur.

> Mitchell et al. recommandent de rapporter les métriques par sous-groupe quand des facteurs démographiques existent. Ce dataset ne contient pas ce type de variables, donc l'analyse par sous-groupe porte sur des facteurs physiques (voir §7).

---

## 5. Evaluation Data

| Champ | Valeur |
|---|---|
| **Dataset** | UCI Concrete Compressive Strength (Yeh, 1998) |
| **Source** | https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength |
| **Observations (après nettoyage)** | 1 005 (25 doublons supprimés sur 1 030 originaux) |
| **Features** | 8 features numériques continues (kg/m³ + jours) |
| **Preprocessing** | Voir Notebook 1 : nettoyage des doublons, vérification des types, conservation des outliers physiquement valides |

### Protocole d'évaluation

**Nested Cross-Validation** :
- **Boucle externe** : `KFold(n_splits=5, shuffle=True, random_state=0)`, donne 5 scores RMSE non biaisés
- **Boucle interne** : `KFold(n_splits=5, shuffle=True, random_state=42)` via `GridSearchCV`, pour le tuning des hyperparamètres
- `shuffle=True` sur les deux boucles car le dataset UCI est ordonné par formulation
- Seeds différents (42 vs 0) pour que les deux boucles ne découpent pas aux mêmes endroits

### Pas de data leakage

Le `StandardScaler` est fitté uniquement sur le train de chaque fold, grâce à l'encapsulation dans un `sklearn.Pipeline`. Les statistiques de normalisation (moyenne, std) ne voient jamais les données de test du fold correspondant.

---

## 6. Training Data

Mêmes données que l'évaluation : le dataset UCI nettoyé (1 005 observations) est utilisé pour entraîner le **modèle final** (Notebook 2, §7), après sélection des hyperparamètres optimaux par GridSearchCV.

Pas de données externes ni d'augmentation de données.

La répartition naturelle du dataset est conservée (pas de stratification, c'est une régression).

---

## 7. Quantitative Analyses

### Résultats par modèle (outer CV, 5 folds)

| Modèle | RMSE moyen | RMSE std | R² moyen | R² std |
|---|---|---|---|---|
| Ridge | 10.385 MPa | 0.449 | 0.590 | 0.031 |
| Random Forest | 4.935 MPa | 0.318 | 0.907 | 0.014 |
| **Gradient Boosting** | **4.208 MPa** | **0.261** | **0.932** | **0.010** |

### Résultats par fold : Gradient Boosting

| Fold | RMSE (MPa) |
|---|---|
| Fold 1 | 4.219 |
| Fold 2 | 4.244 |
| Fold 3 | 3.786 |
| Fold 4 | 4.608 |
| Fold 5 | 4.183 |
| **Moyenne** | **4.208** |
| **Std** | **0.261** |

La faible variance des scores (std = 0.261) montre que l'estimation est stable, pas le fruit d'un fold chanceux.

### Feature Importances : Gradient Boosting (modèle final, fit sur 1 005 observations)

| Feature | Importance relative | Interprétation physique |
|---|---|---|
| `age` | ~35% | Hydratation progressive du ciment, relation logarithmique |
| `cement` | ~29% | Liant principal, corrélation directe avec la résistance |
| `water` | ~11% | Loi eau/ciment de Féret, l'eau en excès crée des pores |
| `slag` | ~8.5% | Liant secondaire à prise lente |
| `superplasticizer` | ~8.3% | Réducteur d'eau, améliore la résistance indirectement |
| `fine_agg` | ~4.5% | Rôle de remplissage |
| `coarse_agg` | ~1.8% | Rôle de remplissage |
| `fly_ash` | ~1.2% | Liant tertiaire à très faible concentration |

### Analyse par sous-groupe physique

| Sous-groupe | Observation |
|---|---|
| **Bétons sans adjuvants** (slag=0, fly_ash=0, SP=0) | Formulations simples, modèle probablement plus précis (zone dense du dataset) |
| **Béton jeune (age < 7j)** | Zone peu couverte, prédictions moins fiables |
| **Béton très résistant (strength > 70 MPa)** | Peu d'observations, risque de sous-estimation |
| **Forte teneur en ciment (cement > 450 kg/m³)** | Zone clairsemée, extrapolation risquée |

### Coefficients Ridge (pour référence, modèle interprétable)

| Feature | Coefficient standardisé |
|---|---|
| `cement` | +12.05 |
| `slag` | +8.40 |
| `age` | +7.13 |
| `fly_ash` | +5.35 |
| `superplasticizer` | +1.68 |
| `fine_agg` | +1.32 |
| `coarse_agg` | +1.10 |
| `water` | **-3.37** |
| Intercept | 35.25 MPa |

Les coefficients Ridge confirment la physique : `cement` est le plus impactant positivement, `water` est le seul négatif (loi de Féret). À prendre avec précaution, Ridge linéarise des relations qui ne le sont pas.

---

## 8. Ethical Considerations

### Données

- Le dataset UCI ne contient aucune donnée personnelle ou sensible, uniquement des mesures physico-chimiques de laboratoire.
- Aucun biais démographique, racial, de genre ou géographique ne s'applique ici.
- Pas de question de fairness au sens classique, le modèle prédit une propriété physique, pas un comportement humain.

### Risques liés à l'usage

| Risque | Niveau | Mitigation |
|---|---|---|
| Décision structurelle basée uniquement sur la prédiction | **Élevé** | Toujours coupler à des essais physiques sur éprouvettes |
| Extrapolation à des formulations hors-domaine | **Moyen** | Vérifier que les inputs sont dans la plage du dataset avant toute prédiction |
| Sous-estimation de résistance, ouvrages sous-dimensionnés | **Élevé** | Appliquer un coefficient de sécurité selon les normes EN 206 |
| Surconfiance dans la précision du modèle | **Moyen** | Toujours reporter le RMSE ± std, pas seulement la valeur prédite |

### Contexte de développement

Ce modèle a été développé dans un cadre **strictement académique** (cours I2ML). Il n'est pas prévu pour un déploiement en production sans validation par des ingénieurs génie civil qualifiés.

---

## 9. Caveats and Recommendations

### Limites identifiées

1. **Extrapolation impossible** : les modèles à base d'arbres prédisent une constante en dehors des plages observées. Toute formulation avec `cement > 540 kg/m³` ou `age > 365j` sera mal estimée.

2. **Biais pessimiste** : les métriques correspondent à des modèles entraînés sur 80% des données. Le modèle final (100% des données) est un peu meilleur, donc le RMSE réel en production est probablement légèrement inférieur à 4.208 MPa.

3. **Dataset de laboratoire** : conditions contrôlées (température, humidité standardisées). La variabilité réelle de chantier n'est pas prise en compte.

4. **Multicolinéarité** : `superplasticizer` et `water` sont corrélés (r ≈ -0.66, EDA). Les Feature Importances de ces deux features peuvent être un peu redistribuées artificiellement.

5. **Limite de l'Impurity Importance** : favorise les features continues à haute cardinalité. Pour une analyse plus robuste, la Permutation Feature Importance serait plus fiable.

6. **Taille du dataset** : 1 005 observations après nettoyage. Un dataset plus grand améliorerait la couverture des zones peu représentées (ages rares, haute teneur en ciment).

### Recommandations

- **Feature engineering** : le ratio `water/cement` est un indicateur classique en génie civil (loi de Féret). L'ajouter explicitement pourrait améliorer les performances de Ridge et aider l'interprétabilité des modèles à arbres.
- **Permutation Importance** : compléter l'analyse Feature Importance par la permutation pour corriger le biais de l'impurity importance.
- **Intervalles de prédiction** : pour un usage ingénierie, il vaudrait mieux reporter un intervalle de prédiction en plus de la valeur ponctuelle (possible avec `GradientBoostingRegressor` en mode quantile ou via bootstrapping).
- **Validation externe** : tester le modèle sur un dataset indépendant pour confirmer qu'il généralise bien au-delà du dataset UCI.
