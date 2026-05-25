# Fiche 2 — Métriques de Régression

> Synthèse 4 — Évaluation de performance

---

## Pourquoi plusieurs métriques ?

Chaque métrique mesure l'erreur différemment. Selon le contexte, certaines sont plus pertinentes.

Dans notre projet : on utilise **RMSE** (principal) + **R²** (complémentaire).

---

## MSE — Mean Squared Error

$$MSE = \frac{1}{m}\sum_{i=1}^m (y_i - \hat{y}_i)^2$$

- Unité : **MPa²** (unité au carré → peu lisible)
- Pénalise **quadratiquement** les grandes erreurs (une erreur de 10 MPa pèse 100× plus qu'une de 1 MPa)
- Utilisé comme **inner loss** dans GridSearchCV (`neg_mean_squared_error`) car différentiable

**Pourquoi on l'utilise en interne** : la descente de gradient et l'optimisation aiment MSE (convexe, dérivable partout).

---

## RMSE — Root Mean Squared Error ← **Notre métrique principale**

$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{m}\sum_{i=1}^m (y_i - \hat{y}_i)^2}$$

- Unité : **MPa** (même unité que la target → directement interprétable)
- **Pourquoi RMSE et pas MSE** : un ingénieur civil raisonne en MPa, pas en MPa². "Mon modèle se trompe en moyenne de 4.2 MPa" a du sens. "Il se trompe de 17.7 MPa²" non.
- Sensible aux outliers (erreurs élevées pèsent beaucoup)

**Notre résultat : GB → RMSE = 4.2 MPa sur une target [2-82 MPa] = erreur relative ~6% ✅**

---

## R² — Coefficient de détermination ← **Notre métrique complémentaire**

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2} = 1 - \frac{SSE_{modèle}}{SSE_{baseline}}$$

- Interprétation : **fraction de variance de y expliquée par le modèle**
- R² = 1 → prédictions parfaites
- R² = 0 → aussi bon qu'un modèle constant (prédire la moyenne)
- R² < 0 → **pire** que de prédire la moyenne (possible sur données de test !)

**Notre résultat : GB → R² = 0.932 → le modèle explique 93.2% de la variance de la résistance ✅**

---

## Comparaison des métriques

| Métrique | Unité | Sensibilité outliers | Utilisé dans notre projet |
|---|---|---|---|
| MSE | MPa² | ⚠️ Forte | Inner loss (GridSearchCV) |
| **RMSE** | **MPa** | ⚠️ Forte | **Outer metric — résultats reportés** |
| MAE | MPa | ✅ Robuste | Non utilisé |
| MAPE | % | Variable | Non utilisé |
| **R²** | Sans unité | ⚠️ Forte | **Complémentaire** |

---

## Pourquoi neg_MSE dans GridSearchCV ?

GridSearchCV **minimise** le score. Sklearn passe la convention que les métriques de "scoring" sont à **maximiser**. Donc on passe `neg_mean_squared_error` (MSE négatif) — GridSearchCV **maximise** -MSE, ce qui **minimise** MSE.

Dans le code :
```python
SCORING_INNER = 'neg_mean_squared_error'
rmse = np.sqrt(-scores)  # on reconvertit en RMSE positif
```

---

## Biais pessimiste des métriques outer CV

$$\mathbb{E}[\widehat{GE}_{CV}] \approx GE(\mathcal{I}, n_{train}) \geq GE(\mathcal{I}, n)$$

Nos RMSE outer CV sont calculés sur des modèles entraînés sur **80% des données** (4 folds sur 5). Le modèle final, entraîné sur 100%, serait légèrement meilleur. L'estimation est **pessimiste** — ce qui est une propriété souhaitable (on ne sur-estime pas).

---

## À retenir pour l'oral

> *"On reporte le RMSE en MPa car c'est directement interprétable : on se trompe en moyenne de 4.2 MPa. On utilise neg_MSE en interne dans GridSearchCV car sklearn maximise les scores. Le R² de 0.93 signifie qu'on explique 93% de la variance de la résistance — un très bon niveau pour ce type de dataset."*
