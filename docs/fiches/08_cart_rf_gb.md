# Fiche 8 — CART, Random Forest & Gradient Boosting

> Synthèse 6 — Arbres CART + Synthèse 7 — Random Forest

---

## CART — La brique de base

### C'est quoi un arbre CART ?

Un arbre de décision qui divise l'espace des features en régions rectangulaires par **splits binaires**.

**Prédiction** : dans chaque feuille, on prédit la **moyenne** des observations :
$$f(x) = \sum_m c_m \cdot \mathbf{1}(x \in Q_m)$$

**Comment il trouve les splits** : à chaque nœud, il cherche le split qui minimise la variance (MSE) dans les deux fils :
$$\arg\min_{j, t} [\mathcal{R}(N_1) + \mathcal{R}(N_2)]$$

Construction **gloutonne** : optimise localement, sans anticiper les splits futurs.

### Pourquoi un seul arbre ne suffit pas ?

| Problème | Conséquence sur le béton |
|---|---|
| **Instabilité (haute variance)** | Un seul outlier de composition peut changer tout l'arbre |
| **Extrapolation impossible** | Prédit une constante par feuille → mauvais hors de la gamme connue |
| **Relations "en escaliers"** | La relation log(age) est approximée par des marches |

### Avantage clé des arbres

**Invariance aux transformations monotones** : un arbre fait des comparaisons (`age > 28 ?`), pas des produits scalaires. `log(age)` et `age` donnent le même arbre. → Le StandardScaler ne change rien pour RF et GB.

---

## Random Forest — Réduire la variance

### Idée centrale : Bagging

Au lieu d'un arbre instable, on en entraîne **M arbres différents** en parallèle et on moyenne leurs prédictions.

Chaque arbre est différent car :
1. **Bootstrap** : entraîné sur un tirage avec remise du dataset
2. **Feature sampling** : à chaque nœud, seules **mtry features** aléatoires sont considérées (mtry ≈ p/3 en régression)

### Pourquoi ça marche ?

$$Var(\hat{f}) = (1-\rho)\frac{\sigma^2}{M} + \rho\sigma^2$$

- M = nombre d'arbres (`n_estimators`) → plus grand = variance réduite
- ρ = corrélation entre arbres → le feature sampling réduit ρ → variance réduite davantage

**RF réduit la variance sans augmenter le biais.**

### Résultats dans notre projet

- **RMSE** : 4.935 ± 0.318 MPa | **R²** : 0.907
- **Meilleurs HPs** : max_depth=20, n_estimators=300, min_samples_leaf=1, min_samples_split=2
- max_depth=20 (pas None) → légère contrainte évite les arbres trop profonds

### Feature Importance RF (impurity-based)

Somme des réductions de MSE aux nœuds où la feature est utilisée, sur tous les arbres.

`age` (35%) > `cement` (29%) > `water` (11%)

**Limite** : favorise les features continues à haute cardinalité (celles qui ont beaucoup de valeurs possibles).

---

## Gradient Boosting — Réduire le biais

### Idée centrale : Boosting séquentiel

Au lieu d'arbres parallèles et indépendants, on entraîne les arbres **en séquence**. Chaque arbre corrige les **résidus** (erreurs) du précédent.

```
Arbre 1 → prédiction 30 MPa (vraie valeur 42 MPa) → résidu = +12
Arbre 2 → apprend le résidu +12 → corrige de +8 → résidu = +4
Arbre 3 → apprend le résidu +4 → corrige de +3 → résidu = +1
...
```

C'est la **descente de gradient** dans l'espace fonctionnel :
$$F_{t+1}(x) = F_t(x) + \alpha \cdot h_t(x)$$

où h_t est un arbre entraîné sur les résidus, et α est le `learning_rate`.

### Hyperparamètres clés

| HP | Rôle | Notre valeur |
|---|---|---|
| `n_estimators` | Nombre d'arbres | 300 |
| `learning_rate` | Contribution de chaque arbre | 0.2 |
| `max_depth` | Profondeur (arbres courts = apprenants faibles) | 4 |
| `subsample` | Stochastic GB (fraction données par arbre) | 1.0 |

**Règle** : learning_rate petit → n_estimators grand nécessaire. Ici learning_rate=0.2 (élevé) + 300 arbres = bon compromis.

### Résultats dans notre projet

- **RMSE** : 4.208 ± 0.261 MPa | **R²** : 0.932
- **Meilleur modèle** des 3

### RF vs GB

| | Random Forest | Gradient Boosting |
|---|---|---|
| **Stratégie** | Parallèle (bagging) | Séquentiel (boosting) |
| **Réduit** | Variance | Biais |
| **Sensibilité aux HPs** | Faible | Plus grande |
| **Vitesse** | Parallélisable | Séquentiel → plus lent |
| **Notre RMSE** | 4.935 MPa | **4.208 MPa** |

---

## À retenir pour l'oral

> *"CART seul est instable — un arbre profond overfit. RF résout ça en faisant la moyenne de 300 arbres bootstrapés (réduit la variance). GB va plus loin en entraînant les arbres séquentiellement pour corriger les erreurs précédentes (réduit le biais). C'est pour ça que GB est légèrement meilleur sur notre dataset — les relations physiques du béton ont un biais résiduel que GB corrige itérativement."*
