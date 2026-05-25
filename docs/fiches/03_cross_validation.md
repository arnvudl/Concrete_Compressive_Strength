# Fiche 3 — Cross-Validation & Biais Pessimiste

> Synthèse 4 — Évaluation de performance

---

## Pourquoi pas juste un train/test split ?

**Problème du hold-out unique** : avec 1 000 observations, si on garde 200 pour le test, on ne sait pas si nos résultats viennent du modèle ou du tirage aléatoire de la partition. Une "mauvaise" partition peut faire varier le RMSE de plusieurs MPa.

**Solution : K-Fold Cross-Validation** → on tourne la partition sur tout le dataset.

---

## K-Fold CV — Mécanisme

```
Dataset : 1 005 observations

Fold 1 : train=[fold2+fold3+fold4+fold5], test=[fold1]  → erreur E1
Fold 2 : train=[fold1+fold3+fold4+fold5], test=[fold2]  → erreur E2
Fold 3 : train=[fold1+fold2+fold4+fold5], test=[fold3]  → erreur E3
Fold 4 : train=[fold1+fold2+fold3+fold5], test=[fold4]  → erreur E4
Fold 5 : train=[fold1+fold2+fold3+fold4], test=[fold5]  → erreur E5

GE_hat = (E1 + E2 + E3 + E4 + E5) / 5
```

Chaque observation est testée **exactement une fois**.  
Les modèles M1…M5 sont des **intermédiaires** → on les jette après.  
Le modèle final = entraîné sur **toutes** les données.

---

## Pourquoi 5-fold dans notre projet ?

Règle empirique de la synthèse 4 :
- n < 200 → LOO ou Repeated CV
- **200 ≤ n ≤ 10⁵ → 5-fold ou 10-fold** ← notre cas (n=1 005)
- n > 10⁵ → hold-out

5-fold = bon compromis biais/variance/coût computationnel.

---

## Biais pessimiste — Propriété fondamentale

$$\mathbb{E}[\widehat{GE}_{CV}] \approx GE(\mathcal{I}, n_{train}) \geq GE(\mathcal{I}, n)$$

**Cause** : à chaque fold, le modèle est entraîné sur **80% des données** (4 folds sur 5), pas sur les 100%.  
**Conséquence** : plus de données = généralement meilleur modèle → le modèle final (100%) sera légèrement meilleur que ce qu'on a estimé.  
**C'est une propriété SOUHAITÉE** : on préfère une estimation pessimiste (prudente) plutôt qu'optimiste.

---

## Non-indépendance des folds

Les k erreurs CV ne sont **pas** indépendantes — les trains sets se chevauchent à k-2 folds en commun.  
→ Faire un t-test classique sur les 5 erreurs CV pour comparer deux modèles est **statistiquement invalide** (Bengio & Grandvalet, 2004).  
→ Il n'existe **aucun estimateur non biaisé** de la variance de l'estimateur CV.

**Dans notre projet** : on compare les modèles sur la moyenne des RMSE outer, pas avec un test statistique formel.

---

## Pourquoi shuffle=True dans notre code ?

```python
outer_cv = KFold(n_splits=5, shuffle=True, random_state=0)
```

**Cause** : le dataset UCI Concrete est ordonné par formulation (pas aléatoirement). Sans shuffle, les folds seraient des blocs séquentiels — fold 1 = formulations 1-201, fold 2 = formulations 202-402, etc. → les folds ne seraient pas représentatifs de la distribution globale.

**Conséquence du shuffle** : chaque fold contient un mélange de toutes les formulations → estimation plus robuste.

---

## Pourquoi des seeds différents ?

```python
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
outer_cv  = KFold(n_splits=5, shuffle=True, random_state=0)
```

**Cause** : si inner et outer utilisent le même seed, ils pourraient créer des partitions structurellement similaires → les folds internes pourraient "toucher" les mêmes données que les folds externes → biais subtil.

**Bonne pratique** : seeds différents = partitions indépendantes = estimation plus robuste.

---

## À retenir pour l'oral

> *"On utilise une 5-fold CV car c'est le standard pour n entre 200 et 100 000. On a shuffle=True car le dataset UCI est ordonné — sans ça, nos folds ne seraient pas représentatifs. Les 5 RMSE qu'on obtient sont légèrement pessimistes car chaque modèle n'a vu que 80% des données — c'est normal et souhaitable."*
