# Fiche 1 — Erreur de Généralisation (GE)

> Synthèse 4 — Évaluation de performance

---

## C'est quoi le problème central ?

Tu entraînes un modèle sur des données. Il fait 0% d'erreur sur ces mêmes données. Est-il bon ?

**Non.** Il a peut-être juste **mémorisé** les réponses.

La vraie question : est-ce qu'il fait de bonnes prédictions sur des données qu'il n'a **jamais vues** ?

---

## Train Error vs GE — La distinction fondamentale

### Train Error (erreur d'entraînement)

$$\mathcal{R}_{emp}(\hat{f}) = \frac{1}{n}\sum_{i=1}^n L(y^{(i)}, \hat{f}(x^{(i)}))$$

**Cause** : on mesure l'erreur sur les mêmes données qu'on a utilisées pour entraîner.  
**Conséquence** : le modèle a été ajusté exprès pour minimiser cette erreur → il "connaît les réponses".  
→ Toujours **optimistement biaisé** : sous-estime l'erreur réelle.

**Cas extrême** : 1-NN → train error = 0 toujours, même si le modèle est nul en généralisation.

---

### Generalization Error (GE)

$$GE(\hat{f}, L) = \mathbb{E}_{(x,y) \sim \mathbb{P}_{xy}}[L(y, \hat{f}(x))]$$

**C'est quoi** : l'erreur moyenne sur des données infinies, tirées du même processus, que le modèle n'a jamais vues.

**Problème** : on ne peut pas la calculer exactement (on n'a pas de données infinies).  
→ On doit **l'estimer** avec des techniques de rééchantillonnage.

---

## Overfitting & Underfitting

| Situation | Train Error | Test Error | Diagnostic |
|---|---|---|---|
| **Underfitting** | Élevé | Élevé | Modèle trop simple — augmenter la complexité |
| **Overfitting** | Faible | Élevé | Modèle trop complexe — régulariser ou plus de données |
| **Bon modèle** | ≈ Test Error | Bas | ✅ |

**Formule** :
- Underfitting = $GE(\hat{f}) - GE(f^*)$ → écart au modèle optimal (biais)
- Overfitting = $GE(\hat{f}) - \mathcal{R}_{emp}(\hat{f})$ → écart train/test (variance)

---

## Décomposition Biais-Variance

$$GE = \underbrace{Biais^2}_{\text{underfitting}} + \underbrace{Variance}_{\text{overfitting}} + \underbrace{\sigma^2_\epsilon}_{\text{bruit irréductible}}$$

- **Biais** = erreur systématique (modèle trop simple → rate la vraie fonction)
- **Variance** = sensibilité aux fluctuations du dataset (modèle trop complexe → overfit le bruit)
- **Bruit** = irréductible, même le modèle parfait fait cette erreur

**Trade-off** : augmenter la complexité → biais ↓ mais variance ↑

---

## Application dans notre projet

- **Ridge** = modèle simple → fort biais (ne capte pas les non-linéarités) → RMSE 10.4 MPa
- **RF/GB** = modèles complexes → biais faible → variance maîtrisée par ensemblage → RMSE ~4-5 MPa
- Le RMSE de 4.2 MPa (GB) **est légèrement pessimiste** : c'est la GE pour un modèle entraîné sur 80% des données. Le modèle final sur 100% est marginalement meilleur.

---

## À retenir pour l'oral

> *"Le train error est biaisé — il mesure la performance sur des données que le modèle a mémorisées. On utilise la cross-validation pour estimer la GE de manière non biaisée. Notre RMSE de 4.2 MPa est calculé sur des folds de test que le modèle n'a jamais vus pendant l'entraînement ou le tuning."*
