# Guide de révision — Défense orale I2ML
## Concrete Compressive Strength — Arnaud & Tim

> Ce document couvre **tous les concepts utilisés dans le projet**, avec les justifications de chaque choix et les questions typiques d'un prof de ML.

---

# 0. La logique globale du projet (à expliquer en premier)

Le projet en une phrase :

> On veut **prédire la résistance à la compression du béton** (en MPa) à partir de 8 ingrédients (ciment, eau, âge...), en utilisant 3 algorithmes de ML différents, évalués de façon rigoureuse et honnête via une **Nested Cross-Validation**.

**Pourquoi c'est une tâche de régression et pas de classification ?**
La cible (`strength`) est une variable continue (2 à 82 MPa). On cherche à prédire une valeur numérique précise, pas une catégorie. Si le prof demandait "résistant / pas résistant" avec un seuil (ex: > 30 MPa = résistant), ça deviendrait de la classification.

**Pourquoi ce dataset ?**
1030 observations, 8 features numériques, zéro valeur manquante, une seule target. C'est propre, ce qui permet de se concentrer sur la méthodologie ML plutôt que sur le nettoyage des données.

---

# 1. Le Data Leakage — LE concept le plus important

## Qu'est-ce que c'est ?

Le data leakage (fuite de données) survient quand **des informations sur les données de test "contaminent" le modèle pendant l'entraînement**. Le modèle "triche" sans qu'on le sache.

## L'exemple concret avec notre projet

Imagine qu'on fait ça **SANS Pipeline** :

```python
# MAUVAIS — data leakage !
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # calcule moyenne/std sur TOUT X
# → le scaler connaît la moyenne/std des données de test
# → ces infos "fuient" dans le modèle via la normalisation

scores = cross_val_score(Ridge(), X_scaled, y, cv=5)
# X_scaled a déjà "vu" les données de test → estimation biaisée
```

Ce qui se passe : le `StandardScaler` calcule la moyenne et l'écart-type sur **toutes les 1005 observations**. Quand la CV ensuite isole un fold de test, ce fold a **déjà influencé** la normalisation. Le modèle connaît indirectement la distribution du test set.

**Conséquence** : le modèle semble meilleur qu'il n'est. En production avec de vraies nouvelles données, les perfs s'effondrent.

## Pourquoi c'est grave en pratique

Dans notre cas béton, imaginer qu'on déploie ce modèle pour optimiser une formulation de chantier. Si l'estimation de la performance est optimiste de 10%, on peut sous-dimensionner une structure. **Un pont peut s'effondrer.**

## La règle absolue

> Tout preprocessing (scaling, imputation, encodage...) doit être **fitté uniquement sur les données d'entraînement de chaque fold**, jamais sur les données de test.

---

# 2. Le Pipeline sklearn — la solution au leakage

## Qu'est-ce que c'est ?

Un Pipeline enchaîne des étapes de preprocessing et un modèle en un seul objet. Quand sklearn appelle `.fit(X_train, y_train)` sur le Pipeline, il :
1. Fitte le StandardScaler sur X_train seulement
2. Transforme X_train
3. Entraîne le modèle sur X_train transformé

Quand sklearn appelle `.predict(X_test)` :
1. Applique la transformation avec les paramètres du train (jamais refitté)
2. Prédit

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),   # étape 1
    ('model', Ridge())              # étape 2
])
```

## Pourquoi c'est obligatoire dans notre projet

Sans Pipeline, dès qu'on met ce code dans une boucle CV, le scaler refittera sur tout le dataset à chaque itération → leakage. Avec le Pipeline, sklearn gère ça automatiquement et correctement.

## Le préfixe `model__`

Pour passer les hyperparamètres d'un modèle dans un Pipeline au GridSearchCV :

```python
param_grid = {
    'model__alpha': [0.001, 0.1, 1, 10, 100]
    # "model" = nom de l'étape dans le Pipeline
    # "alpha" = paramètre de Ridge
}
```

Sans ce préfixe, GridSearchCV ne sait pas à quelle étape du Pipeline l'hyperparamètre appartient.

---

# 3. La Cross-Validation simple — et pourquoi elle ne suffit pas

## Comment ça marche

La k-fold CV divise les données en k blocs (folds). Pour chaque fold :
1. Le fold sert de **test**
2. Les k-1 autres folds servent de **train**
3. On entraîne le modèle et on évalue

On obtient k scores → on fait la moyenne → estimation de la Generalization Error (GE).

**Pourquoi k=5 dans notre projet ?**
Règle empirique : pour n ≈ 1000 observations, 5-fold est le standard. 10-fold serait plus précis mais plus lent, Leave-One-Out serait trop coûteux. 5-fold offre le bon compromis biais/variance/temps.

## Le problème : l'overtuning

Imaginons qu'on teste 72 hyperparamètres (comme notre grille RF) avec une simple CV :

1. Pour chaque combinaison d'HPs, on calcule le score CV
2. On prend le **minimum** (le meilleur score)
3. On annonce ce score comme estimation de performance

**PROBLÈME** : on ne prend pas la moyenne de 72 scores, on prend le **minimum**. Le minimum d'une distribution de 72 valeurs bruitées est systématiquement trop bas (trop optimiste), même si les HPs n'ont aucun effet réel.

**La preuve du cours** (synthèse 10) :
- Classifieur aléatoire, vraie GE = 50% (le modèle tire au hasard)
- On teste 100 configurations d'HPs avec une simple CV
- Score estimé : ~38% → le modèle semble battre le hasard de 12 points
- C'est du **pur cherry-picking** — on a sélectionné la meilleure erreur aléatoire parmi 100

**L'analogie loot box** :
Tu ouvres 100 loot boxes, tu gardes le légendaire +5% stats. Si tu annonces "+5% de stats", tu mens — ce résultat vient du cherry-picking sur 100 tirages, pas d'un vrai avantage.

**Les deux effets qui amplifient le biais** :
- Plus de configurations testées → plus de biais (plus de chances de tomber sur un faux minimum)
- Dataset plus petit → plus de biais (plus de variance par fold → plus facile de "tomber bien")

---

# 4. La Nested Cross-Validation — la solution

## Architecture

```
BOUCLE EXTERNE (outer_cv — 5 folds, random_state=0)
│
│  Pour chaque fold externe :
│  ├─ TEST EXTERNE mis de côté (jamais touché pendant le tuning)
│  │
│  └─ BOUCLE INTERNE (inner_cv — 5 folds, random_state=42) sur le TRAIN EXTERNE
│     │
│     │  Pour chaque combinaison d'HPs :
│     │  ├─ Entraîner le Pipeline sur les 4 folds internes
│     │  └─ Évaluer sur le fold interne restant
│     │
│     └─ → Meilleur λ* sélectionné par GridSearchCV
│
│  ├─ Ré-entraîner avec λ* sur TOUT le train externe
│  └─ Évaluer sur le TEST EXTERNE → score non biaisé
│
└─ 5 scores outer → moyenne = estimation GE non biaisée
```

## Comment ça s'implémente en sklearn

```python
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv  = KFold(n_splits=5, shuffle=True, random_state=0)

search = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='neg_mean_squared_error')
scores = cross_val_score(search, X, y, cv=outer_cv, scoring='neg_mean_squared_error')
```

C'est une seule ligne ! `cross_val_score(GridSearchCV(...))` = nested CV complète.

Quand `cross_val_score` appelle `search.fit(X_train_outer, y_train_outer)`, GridSearchCV relance toute sa propre CV interne sur ce train_outer. La boucle externe ne voit jamais les données internes.

## Pourquoi des random_state différents (42 vs 0) ?

Si les deux boucles utilisent le même seed, elles tendent à couper aux mêmes endroits structurels dans les données. Des seeds différents garantissent que les partitions internes et externes sont vraiment indépendantes → estimation légèrement plus robuste.

## Pourquoi shuffle=True ?

Le dataset UCI est ordonné par formulation (des formulations similaires sont regroupées). Sans shuffle, les folds 1-2-3-4-5 seraient des blocs séquentiels — les premiers folds auraient des bétons similaires en train ET test, créant un biais de structure. Avec shuffle, chaque fold est représentatif de toute la distribution.

## Ce que la nested CV estime exactement

La nested CV estime la performance d'un **algorithme complet** (preprocessing + tuning + entraînement), pas d'un modèle fixe. Elle répond à la question : "Si on donne ce protocole à un nouveau dataset de même taille, quelle performance peut-on espérer ?"

## Le biais pessimiste résiduel

$$\mathbb{E}[\widehat{GE}_{CV}] \approx GE(\mathcal{I}, n_{train}) \geq GE(\mathcal{I}, n)$$

La nested CV estime la GE pour un modèle entraîné sur 80% des données (4 folds sur 5), pas sur 100%. Le modèle final (entraîné sur toutes les données) sera légèrement meilleur. Nos RMSE reportés sont donc **légèrement pessimistes** — c'est une propriété conservatrice et honnête.

---

# 5. GridSearchCV — le tuner

## Qu'est-ce que c'est ?

GridSearchCV teste **toutes les combinaisons possibles** d'une grille d'hyperparamètres, évalue chacune via CV interne, et retourne la meilleure.

```python
param_grid = {'model__alpha': [0.001, 0.1, 1, 10, 100]}
# → 5 valeurs testées × 5 folds inner = 25 entraînements
```

## Pourquoi GridSearch et pas RandomSearch ?

**RandomSearch** (Bergstra & Bengio, 2012) tire aléatoirement des combinaisons d'HPs au lieu de les tester toutes. Il est plus efficace quand :
- L'espace est grand (>3 HPs, ou continu)
- Certains HPs sont peu importants (RandomSearch ne les gaspille pas)

**Dans notre projet**, nos grilles sont **compactes et discrètes** :
- Ridge : 7 valeurs = 7 configs → Grid exhaustif en ≈ 0s
- RF : 72 configs, GB : 72 configs → 2 min en tout sur 32 Go de RAM

Pour des grilles aussi petites, GridSearch explore **tout** l'espace — aucun avantage à Random. Les résultats sont reproductibles et faciles à auditer.

**Et Bayesian Optimization ?**
C'est plus efficace encore pour de très grands espaces (>100 configs). Mais ça ajoute une dépendance (scikit-optimize, optuna) et de la complexité sans gain ici. On a préféré rester simple et lisible.

## Paramètre `refit=True`

Après avoir trouvé le meilleur λ*, GridSearchCV ré-entraîne automatiquement le modèle sur **tout le fold train externe** avec ces HPs. C'est le modèle qu'utilise ensuite `cross_val_score` pour évaluer sur le fold test externe. Sans `refit=True`, on n'aurait pas de modèle prêt à prédire.

---

# 6. Les métriques — MSE, RMSE, R²

## MSE (Mean Squared Error)

$$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

Utilisé comme **scoring interne** dans GridSearchCV (`neg_mean_squared_error`) car il est différentiable et bien adapté à l'optimisation. sklearn le retourne en négatif car il cherche toujours à **maximiser** le score.

**Pourquoi pas MAE ?**
Le MSE punit quadratiquement les grosses erreurs — une erreur de 10 MPa compte 100 fois plus qu'une erreur de 1 MPa. Dans le béton, une grosse erreur de prédiction est disproportionnellement dangereuse → MSE est plus adapté.

## RMSE (Root Mean Squared Error)

$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

On convertit le MSE en RMSE pour le **reporter** car c'est la même unité que la target (MPa). Un RMSE de 4.5 MPa veut dire : en moyenne, la prédiction se trompe de ±4.5 MPa. Un praticien comprend ça directement.

## R² (Coefficient de Détermination)

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

R² mesure la fraction de variance de la target expliquée par le modèle :
- R² = 1 → prédictions parfaites
- R² = 0 → aussi bon qu'un modèle qui prédit toujours la moyenne
- R² < 0 → pire que de prédire la moyenne (modèle cassé)

**Nos résultats** : Ridge R²=0.59, RF R²=0.905, GB R²=0.919 → GB explique 91.9% de la variance de la résistance.

## Pourquoi reporter les deux ?

RMSE = erreur absolue en MPa (utile pour un ingénieur). R² = qualité relative indépendante de l'unité (utile pour comparer des datasets différents). Les deux sont complémentaires.

---

# 7. Ridge Regression — le baseline linéaire

## La formule

$$\hat{\theta} = \arg\min_{\theta} \left[ \underbrace{\frac{1}{n}\sum_{i=1}^n (y_i - \theta^T x_i)^2}_{MSE} + \underbrace{\alpha \sum_{j=1}^p \theta_j^2}_{pénalité L2} \right]$$

C'est une régression linéaire classique (OLS) à laquelle on ajoute une **pénalité L2** sur les coefficients. Plus les coefficients sont grands, plus la pénalité est forte → le modèle est forcé à rester "petit".

## Pourquoi pas OLS pur ?

Le dataset a de la **multicolinéarité** : `superplasticizer` et `water` sont corrélés à -0.66 (vu en EDA). Avec OLS, des features corrélées donnent des coefficients instables et énormes (ils se compensent mutuellement). Ridge stabilise ça en pénalisant les grandes valeurs.

Formellement : OLS peut être écrit comme Ridge avec α=0. Ridge est la version régularisée.

## L'hyperparamètre alpha

| Alpha | Effet |
|---|---|
| 0.001 | Presque OLS — coefficients libres |
| 1 | Régularisation standard (défaut sklearn) |
| 1000 | Forte pénalité → coefficients quasi nuls → modèle quasi constant |

On explore [0.001, 0.01, 0.1, 1, 10, 100, 1000] — une décade par ordre de grandeur, ce qui couvre tout l'espace utile de façon logarithmique.

## Pourquoi pas Lasso ?

**Lasso** = pénalité L1 : $\alpha \sum |\theta_j|$. Il fait de la **sélection automatique de features** (certains coefficients passent à exactement 0). Dans notre cas, on a seulement 8 features — pas besoin de sélection. Ridge suffit.

## Pourquoi StandardScaler est indispensable pour Ridge

`age` varie de 1 à 365 jours. `water` varie de 120 à 247 kg/m³. Sans scaling, le coefficient de `age` serait ~100× plus petit que celui de `water` juste à cause des unités — pas parce que `age` est moins important. La pénalité L2 punirait `water` disproportionnellement.

Avec StandardScaler (moyenne=0, std=1 pour chaque feature), les coefficients sont comparables et la pénalité est équitable.

## Ce que Ridge ne peut pas faire

Les relations béton-résistance sont **non-linéaires** :
- `age` suit une courbe logarithmique (l'hydratation ralentit avec le temps)
- La loi eau/ciment de Féret est exponentielle

Ridge ne peut que tracer des hyperplans dans un espace de 8 dimensions — il approxime ces non-linéarités par une droite. C'est pourquoi on s'attend à ce qu'il soit moins bon que RF et GB. Si Ridge était aussi bon, ça voudrait dire que les relations sont essentiellement linéaires — un résultat intéressant en soi.

---

# 8. Les arbres CART — la brique de base

## Comment ça marche

CART construit un arbre binaire en découpant l'espace des features en régions rectangulaires.

**Pour la régression :**
- Dans chaque nœud : cherche le split $(X_j < t)$ qui minimise la variance des deux groupes créés
- Dans chaque feuille : prédit la **moyenne** des observations

$$f(x) = \sum_{m=1}^{M} c_m \cdot \mathbb{I}(x \in Q_m) \quad \text{avec } c_m = \frac{1}{|N_m|}\sum_{i \in N_m} y_i$$

La recherche du split est **gloutonne** : on optimise localement sans anticiper les splits futurs. C'est efficace mais pas optimal globalement.

## Les limites d'un arbre seul

| Problème | Conséquence sur notre dataset |
|---|---|
| **Instabilité (haute variance)** | Un seul béton outlier peut changer toute la structure de l'arbre |
| **Extrapolation impossible** | Prédit une constante par feuille → mauvais sur des formulations jamais vues |
| **Relations diagonales** | La relation log(age) est approximée par des "escaliers" |

**L'instabilité illustrée** : selon la synthèse 6, "supprimer une seule observation peut changer tout le haut de l'arbre". C'est pour ça qu'on n'utilise jamais CART seul sur un vrai problème de prédiction.

## L'invariance aux transformations monotones

CART ne fait que des comparaisons (`age > 90 ?`), pas des produits scalaires. `log(age)` et `age` donnent exactement le même arbre car l'ordre relatif des observations ne change pas. **C'est pourquoi StandardScaler ne change rien pour RF et GB** — on le garde uniquement pour la cohérence du Pipeline.

---

# 9. Random Forest — réduire la variance

## L'idée fondamentale

Un arbre seul est instable (haute variance). Si on entraîne 200 arbres sur des données légèrement différentes et qu'on fait la moyenne de leurs prédictions, les erreurs aléatoires de chaque arbre s'annulent.

C'est le **Bagging** (Bootstrap Aggregation, Breiman 1996) :
1. Tirer M échantillons bootstrap (avec remise) depuis le dataset
2. Entraîner un arbre sur chaque échantillon
3. Prédiction = moyenne des M arbres

## Le problème du bagging pur

Si tous les arbres utilisent les mêmes features importantes (ex: `cement` est toujours la meilleure feature pour le premier split), les arbres font des erreurs **corrélées**. Moyenner des erreurs corrélées réduit peu la variance.

## La solution RF : le feature sampling (mtry)

À chaque nœud de chaque arbre, on ne considère que `mtry` features tirées au hasard parmi les 8. Si `cement` n'est pas dans le tirage, l'arbre doit se débrouiller avec les autres features → les arbres font des erreurs **différentes** → la moyenne est plus efficace.

## La formule de la variance d'ensemble

$$\text{Var}(\hat{f}) = \underbrace{(1-\rho)\frac{\sigma^2}{M}}_{\text{réduit avec M}} + \underbrace{\rho\sigma^2}_{\text{irréductible}}$$

- $M$ = nombre d'arbres : augmenter M réduit le premier terme
- $\rho$ = corrélation entre arbres : le feature sampling réduit ρ
- $\sigma^2$ = variance d'un arbre seul

**Insight clé** : si ρ=0 (arbres parfaitement décorrélés), la variance → 0 quand M → ∞. Mais ρ ne peut jamais être 0 (les arbres partagent le même dataset). Le feature sampling minimise ρ autant que possible.

## Nos hyperparamètres RF

| HP | Ce qu'on a testé | Pourquoi |
|---|---|---|
| `n_estimators` | [100, 200, 300] | Plus grand = plus stable. Rendement décroissant au-delà. On n'a pas mis 1000 car 300 suffit et ça prend moins de temps |
| `max_depth` | [None, 10, 20, 30] | None = arbres complets. Le bagging compense l'overfitting de chaque arbre seul. Des valeurs limitées testent si des arbres plus courts généralisent mieux |
| `min_samples_split` | [2, 5, 10] | Contrôle la granularité. 2 = splits jusqu'aux feuilles pures. 10 = splits plus conservateurs |
| `min_samples_leaf` | [1, 2] | 1 = feuilles pouvant avoir une seule observation. 2 = plus de lissage |

**72 combinaisons** = 3×4×3×2. On a choisi cette taille car le notebook tourne en ~2 min.

## Pourquoi RF ne nécessite pas de StandardScaler

Les arbres font des comparaisons (`cement > 300 ?`), pas des produits scalaires. `age=365` et `cement=300` ne sont pas mis en rapport numériquement. **Mais on le garde dans le Pipeline pour la cohérence** — comme ça le Pipeline est identique pour tous les modèles.

## RF vs un arbre seul — comparaison rapide

| Critère | CART seul | Random Forest |
|---|---|---|
| Variance | Très haute (instable) | Faible (moyenne de 200 arbres) |
| Biais | Faible (arbres profonds) | Faible (même arbres profonds) |
| Interprétabilité | ✅ Lisible | ❌ 200 arbres = boîte noire |
| Extrapolation | ❌ Impossible | ❌ Impossible (même pb) |
| Overfitting | ❌ Très sensible | ✅ Robuste |

---

# 10. Gradient Boosting — réduire le biais

## L'idée fondamentale

Là où RF entraîne des arbres **en parallèle** (indépendamment), GB les entraîne **en séquence**. Chaque arbre corrige les erreurs (résidus) de tous les arbres précédents.

**Analogie** :
- Arbre 1 prédit la résistance d'un béton : prédit 35 MPa, vraie valeur 45 MPa, erreur = +10 MPa
- Arbre 2 : entraîné pour prédire le résidu (+10 MPa). Il prédit +8 MPa
- Arbre 3 : entraîné pour prédire le résidu restant (+2 MPa)...
- Après 200 arbres : la prédiction converge vers la vraie valeur

## La descente de gradient dans l'espace fonctionnel

$$F_{m}(x) = F_{m-1}(x) + \alpha \cdot h_m(x)$$

- $F_{m-1}$ = modèle après m-1 arbres
- $h_m$ = nouvel arbre entraîné sur les résidus de $F_{m-1}$
- $\alpha$ = learning rate (taux d'apprentissage)

C'est la **descente de gradient** appliquée à une fonction plutôt qu'à des paramètres. On descend pas à pas vers le minimum de la fonction de perte.

## RF réduit la variance, GB réduit le biais

| | Random Forest | Gradient Boosting |
|---|---|---|
| Stratégie | Bagging (parallèle) | Boosting (séquentiel) |
| Ce qu'il réduit | **Variance** | **Biais** |
| Risque principal | Peu (le bagging protège) | Overfitting si mal tuné |
| Speed | Plus rapide (parallélisable) | Plus lent (séquentiel) |
| Performance sur données propres | Bonne | Souvent meilleure |

**Pourquoi GB gagne sur notre dataset ?**
Le béton a des relations physiques complexes mais régulières (loi de Féret, hydratation). Le biais est la principale source d'erreur — GB excelle à réduire le biais. RF réduit la variance mais si les arbres sont tous légèrement biaisés dans la même direction, leur moyenne l'est aussi.

## Nos hyperparamètres GB

| HP | Ce qu'on a testé | Justification |
|---|---|---|
| `n_estimators` | [100, 200, 300] | Nombre d'arbres séquentiels. Plus grand = plus précis |
| `learning_rate` | [0.01, 0.05, 0.1, 0.2] | Contrôle la contribution de chaque arbre. Petit α = descente lente mais stable. Grand α = risque d'overshoot |
| `max_depth` | [3, 4, 5] | En GB, les arbres doivent être **faibles** (peu profonds) ! Si chaque arbre est trop fort, il overfitte sur les résidus → le boosting diverge |
| `subsample` | [0.8, 1.0] | < 1.0 = Stochastic GB : chaque arbre n'est entraîné que sur 80% des données, tirées aléatoirement. Réduit la variance, améliore souvent la GE |

**Lien learning_rate / n_estimators** : un petit learning_rate nécessite plus d'arbres pour converger. `learning_rate=0.1, n_estimators=200` ≈ `learning_rate=0.05, n_estimators=400` en termes de performance. On a testé les deux extrêmes.

**Pourquoi max_depth petit pour GB ?**
En RF, les arbres sont profonds car ils travaillent seuls (le bagging compense l'overfitting). En GB, chaque arbre est un "apprenant faible" qui ne doit corriger qu'une petite partie de l'erreur. Un arbre trop profond en GB capturerait du bruit au lieu d'apprendre la tendance → divergence.

---

# 11. XGBoost — le bonus

## Qu'est-ce qui le différencie du sklearn GB ?

XGBoost (Chen & Guestrin, 2016) est une implémentation optimisée du GB avec :

1. **Régularisation intégrée** : L1 (Lasso) et L2 (Ridge) sur les poids des arbres — un hyperparamètre supplémentaire qui contrôle la complexité
2. **Parallélisation native** : construction des arbres optimisée en C++, beaucoup plus rapide sur de grands datasets
3. **Gestion des NaN native** : apprend automatiquement la direction pour les valeurs manquantes
4. **Early stopping** : peut arrêter l'entraînement quand le score de validation ne s'améliore plus

**Dans notre projet** : XGBoost donne 4.253 ± 0.328 MPa vs GB sklearn 4.584 ± 0.380 MPa (résultats Tim) — statistiquement équivalents (les intervalles se chevauchent). On a conservé sklearn GB car il ne nécessite pas de dépendance externe et est suffisant.

---

# 12. Feature Importance — l'interprétabilité

## Impurity Importance (ce qu'on calcule)

Pour chaque feature $x_j$, on somme toutes les réductions de MSE (en régression) obtenues aux nœuds qui utilisent cette feature, sur tous les arbres.

Une feature utilisée souvent et qui réduit beaucoup le MSE à chaque fois → importance élevée.

**Nos résultats attendus :**
- `cement` : fort impact (liant principal, corrélation +0.50 avec strength)
- `age` : fort impact (hydratation progressive, relation log)
- `water` : impact négatif (loi eau/ciment de Féret)
- `coarse_agg`, `fine_agg` : faible impact (remplissage)

## Permutation Importance (alternative plus robuste)

On permute aléatoirement les valeurs d'une feature (détruit la relation feature-target) et on observe la dégradation du score OOB.

$$\widehat{FI}_j = \widehat{GE}_{OOB, j\text{ permuté}} - \widehat{GE}_{OOB}$$

Plus le score se dégrade, plus la feature est importante.

## Limite commune des deux méthodes

Les deux méthodes ont un **biais en faveur des features continues à haute cardinalité** (Strobl et al. 2007). `cement` (valeurs de 100 à 540) sera toujours légèrement surévalué par rapport à `fly_ash` (60% de valeurs = 0). À mentionner pour montrer la rigueur.

## Pourquoi c'est important pour la défense

Les Feature Importances **confirment la physique du béton**. On ne dit pas juste "le modèle fonctionne bien" — on montre qu'il a appris des lois physiques réelles (loi eau/ciment, hydratation). C'est exactement ce que demande la section Specialized Focus du projet.

---

# 13. Le modèle final

## Pourquoi réentraîner après la nested CV ?

La nested CV donne une **estimation honnête de la performance** — mais les modèles produits sont des modèles intermédiaires, entraînés sur 80% des données. On les jette.

Pour **déployer** le modèle, on réentraîne sur **toutes les données** (1005 observations) en relançant un GridSearchCV pour trouver les meilleurs HPs globaux.

```python
final_search = GridSearchCV(pipe_gb, param_grid_gb, cv=inner_cv, ...)
final_search.fit(X, y)   # tout le dataset
```

**Important** : la performance qu'on annonce reste celle de la nested CV (§6), pas le score inner de ce GridSearch final (qui serait biaisé). Le modèle final prédit en production, la nested CV évalue honnêtement.

---

# 14. Model Card — framework Mitchell et al. (2019)

## Qu'est-ce que c'est ?

Une Model Card est une fiche de documentation standardisée pour un modèle ML. Mitchell et al. (2019) ont proposé ce framework pour rendre les modèles **transparents et auditables**.

## Les sections clés

| Section | Contenu | Pourquoi important |
|---|---|---|
| **Model Details** | Algorithme, auteurs, framework, tâche | Traçabilité |
| **Intended Use** | Usage prévu et usages hors périmètre | Prévention des mauvais usages |
| **Factors** | Features d'entrée, facteurs de risque | Limites de validité |
| **Metrics** | Métriques choisies et valeurs | Transparence sur la performance |
| **Evaluation Data** | Protocole d'évaluation | Reproducibilité |
| **Training Data** | Source des données | Biais potentiels |
| **Ethical Considerations** | Risques d'usage | Responsabilité |
| **Caveats** | Limites identifiées | Honnêteté |

## Pourquoi c'est important pour notre projet

Le projet vaut 15% sur le Reporting Quality. Une Model Card bien faite montre qu'on ne livre pas juste "un modèle qui fonctionne" — on livre un artefact documenté, avec ses limites explicites. C'est la norme en ML industriel depuis 2019.

**Dans notre cas** : le meilleur modèle est Gradient Boosting (RMSE ≈ 4.2-4.6 MPa, R² ≈ 0.919). On documente ses limites (extrapolation impossible, biais pessimiste, dataset de laboratoire) pour qu'un ingénieur civil puisse l'utiliser correctement.

---

# 15. Specialized Focus — interprétabilité en génie civil

## Pourquoi ce focus ?

Le projet demande (15% des points) : réfléchir à ce qui est **spécifiquement important** pour ce dataset et ce domaine d'application, et montrer que nos choix ont été influencés par ça.

## Le domaine de la construction a des contraintes spécifiques

1. **Sécurité critique** : un modèle utilisé pour dimensionner une structure doit être explicable. Si le modèle dit "cette formulation tient à 40 MPa" et que l'ingénieur ne peut pas comprendre pourquoi, il ne peut pas valider la prédiction.

2. **Règlementaire** : les normes de béton (EN 206, Eurocode 2) exigent des justifications physiques. "Le modèle de ML le dit" n'est pas une justification légale.

3. **Double besoin** : prédiction (quelle résistance ?) + explication (quel ingrédient ajuster ?).

## Comment ça a influencé nos choix

**On a exclu les réseaux de neurones** précisément à cause de l'interprétabilité. Un réseau à 3 couches cachées et 500 neurones prédit bien mais ne peut pas expliquer pourquoi. En génie civil, c'est inacceptable.

**On a choisi RMSE (MPa) comme métrique principale** (et pas juste R²) car les ingénieurs raisonnent en MPa — c'est leur unité de travail quotidienne.

**Les Feature Importances confirment la physique** : cement et age dominent, water a un impact négatif — cohérent avec la loi eau/ciment de Féret (1897). Un modèle qui "apprend" les bonnes lois physiques est plus fiable qu'un modèle qui donne juste un bon score.

---

# 16. Le biais-variance tradeoff

## La formule fondamentale

$$GE = Biais^2 + Variance + Bruit$$

- **Biais** : erreur systématique. Un modèle trop simple (underfitting) a un biais élevé.
- **Variance** : sensibilité aux données d'entraînement. Un modèle trop complexe (overfitting) a une variance élevée.
- **Bruit** : irréductible, vient des données elles-mêmes.

## Comment nos 3 modèles se situent

| Modèle | Biais | Variance | Commentaire |
|---|---|---|---|
| Ridge | Élevé | Faible | Modèle linéaire → ne capte pas les non-linéarités |
| Random Forest | Faible | Faible | Arbres profonds (faible biais) + bagging (réduit variance) |
| Gradient Boosting | Très faible | Modéré | Boosting séquentiel réduit le biais, mais plus sensible à l'overfitting |

## Est-ce qu'il y a de l'overfitting dans notre projet ?

Non, selon les résultats. L'écart entre le RMSE inner CV (optimiste) et le RMSE outer CV (honnête) n'est pas énorme :
- GB : inner ≈ 4.4 MPa, outer ≈ 4.6 MPa → faible overfitting
- RF : inner ≈ 4.9 MPa, outer ≈ 5.0 MPa → quasi pas d'overfitting

Si le RMSE outer était 8 MPa alors que le inner était 2 MPa, ça signifierait que le modèle overfitte massivemement sur les folds d'entraînement.

---

# 17. Questions typiques du prof — avec les réponses

## "Pourquoi la nested CV et pas une simple CV ?"

Sans nested CV, les HPs sont tunés sur les mêmes données qui servent à l'évaluation. On sélectionne le minimum d'une distribution bruitée → biais optimiste (synthèse 10). La nested CV sépare physiquement le tuning de l'évaluation : la boucle interne tune, la boucle externe évalue sur des données que l'inner n'a jamais vues.

## "Comment le Pipeline évite le data leakage ?"

Le StandardScaler est fitté uniquement sur `X_train` de chaque fold, jamais sur `X_test`. Sans Pipeline, si on scale avant la CV, les statistiques (moyenne, std) sont calculées sur toutes les données → fuite d'information sur le test set. sklearn gère ça automatiquement dans un Pipeline.

## "Pourquoi StandardScaler pour RF et GB ?"

Techniquement pas nécessaire — les arbres font des comparaisons, pas des produits scalaires. Mais on le garde car (1) le Pipeline doit être cohérent entre tous les modèles, (2) ça ne nuit pas aux arbres, (3) si on ajoutait un Ridge dans un futur comparatif, il serait déjà protégé.

## "Pourquoi GridSearch et pas Random Search ?"

Nos grilles sont compactes (≤ 72 combinaisons). Grid explore tout l'espace exhaustivement. Random Search serait avantageux pour de très grands espaces continus (>100 configs) — ce n'est pas notre cas.

## "Gradient Boosting est-il en train d'overfitter ?"

Le biais pessimiste de la nested CV est faible (inner vs outer RMSE proche). Les max_depth petits (3-5) et le learning_rate modéré (0.05-0.1) contrôlent la complexité de chaque arbre. Le subsample=0.8 (Stochastic GB) ajoute une régularisation supplémentaire.

## "Pourquoi des seeds différents pour inner et outer CV ?"

Si les deux boucles utilisent le même seed, elles tendent à couper aux mêmes endroits structurels dans les données (les observations ordonnées de façon similaire tomberaient dans les mêmes groupes). Des seeds différents (42 vs 0) garantissent l'indépendance structurelle des deux boucles.

## "Pourquoi le modèle final est-il réentraîné sur tout le dataset ?"

La nested CV donne des estimations honnêtes de performance, mais les modèles produits sont intermédiaires (entraînés sur 80% des données). Pour déployer, on réentraîne sur 100% des données avec le meilleur λ* — le modèle a vu plus de données → légèrement meilleur. La performance annoncée reste celle de la nested CV (honnête).

## "Comment interpréter un RMSE de 4.6 MPa ?"

La target varie de 2 à 82 MPa (range = 80 MPa). Un RMSE de 4.6 MPa = erreur relative d'environ 5-6% sur le range. Les normes béton (EN 206) acceptent des variations de ±5 MPa pour des bétons de classe C30. On est dans l'ordre de grandeur de la précision industrielle.

## "Pourquoi vous n'avez pas utilisé de réseau de neurones ?"

Pour deux raisons : (1) interprétabilité — dans le génie civil, l'ingénieur doit pouvoir expliquer ses choix de formulation, un réseau à 500 neurones ne le permet pas. (2) taille du dataset — 1005 observations, c'est petit pour un réseau de neurones. RF et GB sont généralement supérieurs sur des datasets de cette taille.

## "Qu'est-ce qui est le plus important dans le béton selon vos Feature Importances ?"

`cement` et `age` dominent. Physiquement cohérent : le ciment est le liant principal (plus on en met, plus le béton résiste), l'âge reflète l'hydratation progressive (le béton durcit pendant des semaines). `water` a un impact négatif via la loi eau/ciment de Féret : plus d'eau dilue la structure et crée des pores.

## "La différence entre RF et GB est-elle significative ?"

Sur les résultats d'Arnaud (outer CV) : RF=4.99 MPa, GB=4.58 MPa, écart ≈ 0.4 MPa. Les barres d'erreur se chevauchent légèrement. La différence n'est pas énorme mais GB est consistamment meilleur sur les 5 folds. Sur les résultats de Tim (grilles plus larges) : RF=5.08, GB=4.20, écart plus marqué. On choisit GB mais on peut noter que RF serait un choix défendable pour sa robustesse et interprétabilité.

---

# 18. Tableau récapitulatif des choix et justifications

| Choix | Alternative non retenue | Pourquoi notre choix |
|---|---|---|
| Pipeline sklearn | Preprocessing hors Pipeline | Évite le data leakage — non négociable |
| Nested CV | Simple CV | Sans nested : biais optimiste du score final |
| GridSearchCV | RandomSearch, Bayesian | Grilles compactes → Grid exhaustif suffisant |
| 5 folds inner + 5 folds outer | 3×5 ou 10×10 | Symétrique, standard pour n≈1000 |
| Seeds différents (42 vs 0) | Même seed | Indépendance structurelle des boucles |
| RMSE comme métrique principale | R² seul | RMSE en MPa = interprétable par un ingénieur |
| neg_MSE pour scoring inner | neg_RMSE | Numériquement stable, différentiable |
| 3 algorithmes distincts | Un seul très tuné | Exigence du projet + comparaison biais/variance |
| StandardScaler dans Pipeline | MinMaxScaler, pas de scaling | Adapté à Ridge + cohérence Pipeline |
| Réseaux de neurones exclus | Inclus pour comparaison | Interprétabilité critique en génie civil |
| XGBoost comme bonus | Inclus comme modèle principal | Dépendance externe pas nécessaire ici |
| Ridge comme baseline | OLS pur | Multicolinéarité → Ridge plus robuste |

---

# 19. Les formules clés à retenir

$$\hat{\theta}_{Ridge} = \arg\min_\theta \left[\frac{1}{n}\sum(y_i - \theta^T x_i)^2 + \alpha\sum\theta_j^2\right]$$

$$f_{CART}(x) = \sum_{m=1}^M c_m \cdot \mathbb{I}(x \in Q_m)$$

$$\hat{f}_{RF}(x) = \frac{1}{M}\sum_{m=1}^M \hat{f}^{[m]}(x)$$

$$\text{Var}(\hat{f}_{RF}) = (1-\rho)\frac{\sigma^2}{M} + \rho\sigma^2$$

$$F_m(x) = F_{m-1}(x) + \alpha \cdot h_m(x) \quad \text{(GB)}$$

$$RMSE = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

$$\mathbb{E}[\widehat{GE}_{CV}] \approx GE(\mathcal{I}, n_{train}) \geq GE(\mathcal{I}, n) \quad \text{(biais pessimiste)}$$

---

*Document de révision — Projet I2ML 2026 — Arnaud & Tim*
