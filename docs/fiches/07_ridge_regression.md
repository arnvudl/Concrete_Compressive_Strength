# Fiche 7 — Ridge Regression (Modèle 1 — Baseline)

> Synthèse 2 — Régression Supervisée

---

## C'est quoi Ridge ?

Ridge = régression linéaire + pénalité L2.

$$\hat{\theta} = \arg\min_{\theta} \left[ \underbrace{\frac{1}{n}\sum_{i=1}^n (y_i - \theta^T x_i)^2}_{MSE} + \underbrace{\alpha \sum_{j=1}^p \theta_j^2}_{\text{pénalité L2}} \right]$$

- Si `alpha = 0` → OLS pure (pas de régularisation)
- Si `alpha → ∞` → tous les coefficients → 0 (modèle constant)
- **`alpha` optimal** (notre projet) : **alpha = 1** trouvé par GridSearchCV

---

## Pourquoi Ridge et pas OLS ?

Le dataset Concrete présente de la **multicolinéarité** :  
`superplasticizer` et `water` sont corrélés à r ≈ -0.66.

Avec OLS, la multicolinéarité rend les coefficients instables (grande variance). Ridge stabilise les coefficients via la pénalité L2.

---

## Pourquoi le StandardScaler est OBLIGATOIRE pour Ridge

**Sans scaling** : `age` varie de 1 à 365, `water` varie de 122 à 247. Si `age` a un coefficient θ = 0.2 et `water` un coefficient θ = 5, ce n'est pas parce qu'`age` est moins important — c'est parce que son échelle est différente.

**La pénalité L2 pénalise les grands coefficients**. Sans scaling, les features à petite échelle (comme `age`) auraient des θ plus petits et seraient moins pénalisées → inéquité.

**Avec StandardScaler** : toutes les features sont ramenées à moyenne=0, std=1 → les coefficients sont directement comparables → la pénalité L2 est équitable.

---

## Interprétation des coefficients standardisés

| Feature | Coefficient | Interprétation |
|---|---|---|
| `cement` | **+12.05** | Le plus impactant — +1 std de ciment ↑ résistance de 12 MPa |
| `slag` | +8.40 | Liant secondaire |
| `age` | +7.13 | Hydratation progressive |
| `fly_ash` | +5.35 | Liant tertiaire |
| `superplasticizer` | +1.68 | Réducteur d'eau |
| `fine_agg` | +1.32 | Remplissage |
| `coarse_agg` | +1.10 | Remplissage |
| `water` | **-3.37** | Le seul négatif — loi de Féret |
| Intercept | 35.25 MPa | Résistance moyenne de base |

**Avantage de Ridge** : c'est le seul de nos 3 modèles à donner des coefficients directement interprétables avec direction (+ ou -).

---

## Résultats dans notre projet

- **RMSE outer CV** : 10.385 ± 0.449 MPa
- **R²** : 0.590
- **Rang** : 3/3 — le moins performant

**Pourquoi Ridge est moins bon** : les relations béton-résistance sont **non-linéaires** :
- `age` suit une courbe logarithmique (hydratation) — Ridge l'approxime par une droite
- La loi eau/ciment est quadratique — Ridge ne peut pas la capter

**Ridge est notre baseline** : si les modèles non-linéaires ne faisaient pas mieux, cela suggérerait que les relations sont principalement linéaires.

---

## Pourquoi Ridge comme modèle 1 ?

> "Le modèle le plus simple d'abord" — principe de parcimonie (Ockham)

Ridge est interprétable, rapide, et sert de référence. Si GB fait 4.2 MPa et Ridge 10.4 MPa, on peut quantifier le gain de la non-linéarité.

---

## À retenir pour l'oral

> *"Ridge est notre baseline linéaire. Il ajoute une pénalité L2 à OLS pour gérer la multicolinéarité du dataset. Le StandardScaler est obligatoire avant Ridge pour que la pénalité soit équitable entre les features. Son RMSE de 10.4 MPa confirme que les relations sont fortement non-linéaires — ce que RF et GB capturent bien mieux."*
