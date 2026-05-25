# Fiche 0 — Le Projet & Le Dataset

---

## C'est quoi le projet ?

On veut **prédire la résistance à la compression du béton** (en MPa) à partir de sa composition chimique et de son âge. C'est un problème de **régression supervisée** : on a des données étiquetées (on connaît la vraie résistance mesurée en labo), et on veut apprendre à prédire cette valeur.

---

## Le dataset : UCI Concrete Compressive Strength

| Info | Valeur |
|---|---|
| **Source** | UCI Machine Learning Repository, Yeh (1998) |
| **Taille originale** | 1 030 observations |
| **Taille après nettoyage** | 1 005 observations (25 doublons supprimés) |
| **Features** | 8 variables numériques continues |
| **Target** | `strength` — résistance à la compression (MPa) |

---

## Les 8 features

| Feature | Unité | Ce que c'est |
|---|---|---|
| `cement` | kg/m³ | Liant principal — réagit avec l'eau (hydratation) pour durcir |
| `slag` | kg/m³ | Laitier de haut-fourneau — substitut partiel du ciment, prise lente |
| `fly_ash` | kg/m³ | Cendres volantes — autre substitut, très long terme |
| `water` | kg/m³ | L'eau déclenche l'hydratation — en excès, crée des pores → fragilise |
| `superplasticizer` | kg/m³ | Adjuvant chimique — réduit la quantité d'eau nécessaire |
| `coarse_agg` | kg/m³ | Granulats grossiers (gravier) — rôle de remplissage |
| `fine_agg` | kg/m³ | Granulats fins (sable) — rôle de remplissage |
| `age` | jours | Age du béton au moment du test (1 à 365 jours) |

---

## La target : `strength` (MPa)

- Moyenne : **~35 MPa**, std : **~16 MPa**, range : **[2, 82 MPa]**
- Distribution légèrement asymétrique à droite (skewness > 0)
- Un béton "standard" pour la construction = 25-40 MPa
- Béton haute performance = 60+ MPa

---

## Pourquoi c'est difficile à prédire ?

1. **Relations non-linéaires** : l'âge suit une loi logarithmique d'hydratation (durcit vite au début, lentement ensuite). Ridge (linéaire) ne peut pas capter ça.
2. **Interactions entre features** : le ratio eau/ciment (loi de Féret) est plus important que l'eau ou le ciment séparément.
3. **Zéros nombreux** : slag, fly_ash, superplasticizer = 0 dans 40-70% des cas (bétons sans ces adjuvants).
4. **Multicolinéarité** : superplasticizer et water corrélés à r ≈ -0.66.

---

## Physique clé à retenir pour l'oral

**Loi de Féret (1897)** : résistance ∝ 1 / (ratio eau/ciment)²  
→ Plus il y a d'eau par rapport au ciment, plus le béton est poreux et fragile.

**Hydratation du ciment** : la résistance augmente en log(âge)  
→ Un béton continue de durcir pendant des semaines après la coulée.

**Ciment = liant principal** : corrélation r ≈ +0.50 avec la résistance (la plus forte du dataset).

---

## Ce qu'on a fait dans le projet

1. **Notebook 1** : EDA + nettoyage → export `concrete_clean.csv`
2. **Notebook 2** : Nested CV avec 3 modèles (Ridge, RF, GB) + Specialized Focus + Model Card

**Résultats finaux (nested CV 5-fold) :**

| Modèle | RMSE | R² |
|---|---|---|
| Ridge | 10.4 MPa | 0.59 |
| Random Forest | 4.9 MPa | 0.91 |
| **Gradient Boosting** | **4.2 MPa** | **0.93** |
