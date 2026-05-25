# Fiche 5 — Nested Cross-Validation

> Synthèse 10 — Nested Resampling + Synthèse 9 — Tuning

---

## Le problème : pourquoi une simple CV ne suffit pas ?

**Situation** : tu tunes 50 valeurs d'alpha avec une 5-fold CV. Tu gardes l'alpha qui a le meilleur score.

**Cause du problème** : tu ne prends pas la **moyenne** des 50 scores, tu prends le **minimum**. Le minimum d'une distribution bruitée est toujours **trop bas** (trop optimiste).

**Preuve par l'absurde** (synthèse 10) : classifieur binaire **aléatoire** (GE réelle = 50%).
- 1 run CV → score ≈ 50% ✅
- Avec 100 configs tunées, le "meilleur" score descend à ~38% 🚨
- Le modèle semble meilleur de 12 points alors qu'il est **purement aléatoire**

**Plus de configs testées + dataset plus petit = biais plus grand**

---

## L'analogie de la loot box (synthèse 10)

Tu ouvres 100 loot boxes dans un jeu. Par chance, une te donne +5% de stats.  
Si tu annonces "+5% de stats", tu **mens** — c'est du cherry-picking sur 100 tirages, pas de la vraie puissance.

→ Tuner sans test set séparé, c'est exactement ça.

---

## La solution : deux boucles séparées

```
BOUCLE EXTERNE (outer_cv — 5 folds)
│
│  Pour chaque fold externe :
│  ├─ Isoler le TEST EXTERNE → jamais touché pendant le tuning
│  │
│  └─ BOUCLE INTERNE (inner_cv — 5 folds) sur TRAIN EXTERNE
│     ├─ Tester chaque combinaison d'HPs via GridSearchCV
│     └─ Garder le meilleur λ*
│
│  ├─ Ré-entraîner avec λ* sur tout le TRAIN EXTERNE
│  └─ Évaluer sur TEST EXTERNE → score non biaisé
│
└─ RÉSULTAT : 5 scores outer → moyenne = GE estimée
```

**Garantie** : le test externe n'a **jamais** participé au tuning interne → score non biaisé.

---

## Implémentation sklearn — Une seule ligne

```python
cross_val_score(           # ← BOUCLE EXTERNE
    estimator = GridSearchCV(  # ← BOUCLE INTERNE
        pipe,
        param_grid,
        cv = inner_cv      # KFold(5, shuffle=True, random_state=42)
    ),
    cv = outer_cv          # KFold(5, shuffle=True, random_state=0)
)
```

`GridSearchCV` est passé comme **estimateur** à `cross_val_score` → sklearn gère tout automatiquement.

---

## Coût computationnel

Pour 1 modèle (ex. RF, 72 combos) :
- 5 folds outer × 5 folds inner × 72 combos = **1 800 entraînements**

C'est pour ça qu'on garde des grilles compactes (72 combos max) et qu'on utilise `n_jobs=-1` (parallélisation).

---

## Preuve que ça marche (synthèse 10)

- **Sans nested CV** : le score "tuned" descend vers 0.30-0.45 pour un classifieur aléatoire
- **Avec nested CV** : le score reste stable autour de **0.50** (la vraie GE) — quelle que soit la quantité de configs testées

Le nested CV corrige le biais. Il ne dit pas que le modèle est bon — il dit **honnêtement** ce qu'il vaut.

---

## Réinterprétation architecturale (synthèse 10)

Le processus `[Run CV interne → sélectionner λ* → ré-entraîner]` est un **algorithme self-tuning**.  
Ses inputs = les données brutes.  
Son output = un modèle.  
Les HPs ont **disparu de l'interface visible** — résolus en interne.

La boucle externe évalue **cet algorithme complet**, pas juste le modèle final.

---

## Simple CV vs Nested CV

| Critère | CV simple | Nested CV |
|---|---|---|
| **Biais GE** | ❌ Optimiste (croît avec #configs) | ✅ Non biaisé |
| **Variance** | ⚠️ Modérée | ✅ Réduite |
| **Coût** | Faible | ❌ k_outer × k_inner × #configs |
| **Petits datasets** | ❌ Biais fort | ✅ Conçu pour ça |

---

## À retenir pour l'oral

> *"La simple CV ne suffit pas pour évaluer un modèle après tuning : on sélectionne le meilleur score parmi plusieurs évaluations, ce qui est optimistement biaisé. La nested CV résout ce problème en séparant strictement la boucle de tuning (inner) de la boucle d'évaluation (outer). Dans sklearn, c'est `cross_val_score(GridSearchCV(...))` — une seule ligne qui gère les deux boucles automatiquement."*
