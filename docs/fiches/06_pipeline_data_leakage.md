# Fiche 6 — Pipeline sklearn & Data Leakage

> Synthèse 9 — Tuning (section Pipelines ML)

---

## C'est quoi le data leakage ?

**Définition** : le modèle "voit" indirectement des informations du test set pendant l'entraînement.

**Conséquence** : le modèle semble plus performant qu'il ne l'est réellement → fausse évaluation.

---

## Le piège classique du StandardScaler

### ❌ MAUVAISE pratique

```python
# On scale AVANT la CV → LEAKAGE !
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # utilise toutes les données !

scores = cross_val_score(Ridge(), X_scaled, y, cv=5)
```

**Pourquoi c'est du leakage** :  
`fit_transform(X)` calcule la moyenne et l'écart-type sur **tout X** (y compris les observations qui seront dans le fold de test). Ces statistiques "contaminent" le test set — le scaler a appris quelque chose sur des données qu'il ne devrait pas voir.

---

### ✅ BONNE pratique avec Pipeline

```python
pipe = Pipeline([
    ('scaler', StandardScaler()),  # fitté uniquement sur le train du fold
    ('model', Ridge())
])

scores = cross_val_score(pipe, X, y, cv=5)
```

**Comment sklearn gère ça** :  
Pour chaque fold, sklearn appelle :
1. `pipe.fit(X_train_fold)` → le scaler apprend moyenne/std sur le **train du fold uniquement**
2. `pipe.predict(X_test_fold)` → le scaler transforme le test avec les stats du train

Le test set ne contribue jamais au calcul des statistiques de normalisation.

---

## La règle fondamentale

> **Tout preprocessing qui "apprend" quelque chose des données doit être DANS le Pipeline, JAMAIS avant.**

Ceci inclut : StandardScaler, MinMaxScaler, imputation de valeurs manquantes, encodage basé sur les fréquences, sélection de features, PCA.

---

## Pourquoi StandardScaler dans notre projet même pour RF/GB ?

Techniquement, les arbres (RF, GB) **n'ont pas besoin** de scaling — ils font des comparaisons (`cement > 300 ?`), pas des produits scalaires. Scaling ne change pas les splits.

Mais on le garde pour :
1. **Cohérence** : même Pipeline pour tous les modèles → comparaison équitable
2. **Ridge l'exige** : sans scaling, les coefficients de Ridge seraient biaisés (features à grande échelle dominent la pénalité L2)
3. **Bonne pratique** : Pipeline complet même si inutile pour certains modèles

---

## Le prefixe `model__` — Accès aux HPs dans un Pipeline

```python
# Pipeline : scaler → model
pipe = Pipeline([('scaler', StandardScaler()), ('model', Ridge())])

# Pour accéder aux HPs du modèle :
param_grid = {'model__alpha': [0.1, 1, 10]}
#              ^^^^^^ nom de l'étape dans le Pipeline
```

**Cause** : le Pipeline a plusieurs étapes. Il faut spécifier à quelle étape chaque HP appartient.  
**Format** : `nom_etape__nom_parametre`

---

## Pipeline séquentiel dans notre projet

```
Données brutes → [StandardScaler] → [Ridge / RF / GB] → Prédictions
                   fit sur train          fit sur train
                   transform train        predict test
                   transform test
```

Chaque composant du Pipeline a une phase `.fit()` et une phase `.transform()`/`.predict()`. Pendant la CV, sklearn appelle `.fit()` uniquement sur le train du fold.

---

## Résumé du mécanisme anti-leakage

| Étape | Sans Pipeline (❌) | Avec Pipeline (✅) |
|---|---|---|
| Calcul moyenne/std | Sur tout X | Sur train du fold uniquement |
| Transform du test set | Avec stats de tout X (leak) | Avec stats du train du fold |
| GE estimée | Optimistement biaisée | Non biaisée |

---

## À retenir pour l'oral

> *"Sans Pipeline, si on scale avant la CV, le StandardScaler a vu les données de test — c'est du data leakage. Avec Pipeline, sklearn appelle .fit() uniquement sur le train de chaque fold. C'est garanti par construction. Le préfixe `model__` permet d'accéder aux HPs du modèle malgré l'encapsulation dans le Pipeline."*
