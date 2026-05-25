# Fiche 4 — Hyperparamètres & Tuning

> Synthèse 9 — Tuning

---

## Paramètres θ vs Hyperparamètres λ

| | Paramètres θ | Hyperparamètres λ |
|---|---|---|
| **Définition** | Appris automatiquement pendant l'entraînement | Fixés AVANT l'entraînement |
| **Exemples** | Coefficients Ridge, poids d'un réseau | alpha de Ridge, n_estimators de RF |
| **Comment les trouver** | Minimisation de la loss | Tuning (GridSearch, RandomSearch…) |
| **Sklearn** | Appris par `.fit()` | Passés en argument au constructeur |

---

## Pourquoi le tuning est difficile ?

4 propriétés fondamentales (synthèse 9) :

| Propriété | Problème |
|---|---|
| **Boîte noire** | Pas de gradient ∇c(λ) disponible — on ne peut pas dériver |
| **Coût élevé** | Chaque test = entraîner le modèle entier |
| **Stochasticité** | Le score varie selon la partition du resampling |
| **Structure complexe** | HPs dépendants entre eux (hiérarchique) |

---

## Grid Search vs Random Search

### Grid Search
Teste **toutes les combinaisons** d'une grille discrète.

| RF | n_estimators=[100, 200, 300] | max_depth=[None, 10, 20, 30] | → 3×4=12 |
Avec min_samples_split et min_samples_leaf → 3×4×3×2 = **72 combinaisons**

**Avantage** : couvre tout exhaustivement.  
**Limite** : exponentiel en nombre de HPs.

### Random Search
Échantillonne aléatoirement dans l'espace continu.

**Avantage clé** : si seul x1 influence vraiment la performance, Grid 5×5 = 5 valeurs de x1 testées. Random 25 évals = **25 valeurs distinctes** de x1.

**Pourquoi on a choisi Grid Search** : nos grilles sont compactes (≤72 combinaisons) et discrètes. GridSearch explore tout exhaustivement, résultats reproductibles.

---

## Nos hyperparamètres — Justification

### Ridge : `alpha`
```python
'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
```
- alpha contrôle la force de la régularisation L2
- petit → proche OLS → risque overfitting
- grand → coefficients très petits → risque underfitting
- **Meilleur : alpha=1** (régularisation standard sklearn)

### Random Forest
```python
'model__n_estimators': [100, 200, 300]        # plus = plus stable
'model__max_depth': [None, 10, 20, 30]        # None = arbres complets
'model__min_samples_split': [2, 5, 10]        # granularité des splits
'model__min_samples_leaf': [1, 2]             # taille minimale des feuilles
```
- **Meilleur : max_depth=20, n_estimators=300, min_samples_leaf=1, min_samples_split=2**
- max_depth=None ou 20 → le bagging compense l'overfitting de chaque arbre

### Gradient Boosting
```python
'model__n_estimators': [100, 200, 300]        # nombre d'arbres séquentiels
'model__learning_rate': [0.01, 0.05, 0.1, 0.2] # α de la descente de gradient
'model__max_depth': [3, 4, 5]                 # arbres courts = apprenants faibles
'model__subsample': [0.8, 1.0]               # Stochastic GB
```
- **Meilleur : learning_rate=0.2, max_depth=4, n_estimators=300, subsample=1.0**
- max_depth petit (3-5) = recommandé pour GB → chaque arbre est un "apprenant faible"

---

## Le préfixe `model__` dans sklearn Pipeline

```python
# ❌ Sans pipeline
param_grid = {'alpha': [0.1, 1, 10]}

# ✅ Avec Pipeline
param_grid = {'model__alpha': [0.1, 1, 10]}
```

**Cause** : le Pipeline a plusieurs étapes (`scaler`, `model`). Il faut préciser à quelle étape appartient chaque HP. Le préfixe = nom de l'étape + `__` + nom du paramètre.

---

## Le problème de l'overtuning (synthèse 10)

**Cause** : si on tune sur les mêmes données qu'on évalue, on sélectionne le **minimum** d'une distribution bruitée. Ce minimum est **toujours trop optimiste**.

**Exemple** : classifieur aléatoire (GE réelle = 50%), 100 configs testées → le "meilleur" score CV descend à 38% → on croit avoir un bon modèle alors qu'il est aléatoire.

**Plus on teste de configs, plus le biais est grand.** C'est pour ça qu'il faut la **nested CV**.

---

## À retenir pour l'oral

> *"Les hyperparamètres ne sont pas appris par le modèle — on doit les fixer avant. On utilise GridSearchCV car nos grilles sont compactes. Le préfixe `model__` est obligatoire dans un Pipeline sklearn pour cibler la bonne étape. On a choisi des grilles raisonnables : 72 combos pour RF et GB, ce qui représente ~1800 entraînements par modèle avec la nested CV."*
