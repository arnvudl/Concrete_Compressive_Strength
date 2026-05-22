# Model Card — Concrete Compressive Strength Predictor

> Suivant le framework Mitchell et al. (2019) — *Model Cards for Model Reporting*

---

## 1. Model Details

| Champ | Valeur |
|---|---|
| **Développeurs** | Tim & Arnaud — HELMo BLOC 2, cours I2ML |
| **Date** | Mai 2026 |
| **Type de modèle** | Gradient Boosting Regressor (sklearn) |
| **Version** | 1.0 |
| **Tâche** | Régression supervisée |
| **Hyperparamètres retenus** | `n_estimators=300`, `learning_rate=0.2`, `max_depth=3`, `subsample=0.8` |
| **Preprocessing** | StandardScaler (encapsulé dans un sklearn Pipeline) |
| **Lien repo** | https://github.com/arnvudl/Concrete_Compressive_Strength |

---

## 2. Intended Use

### Usage prévu
Prédire la résistance à la compression (en MPa) d'un béton à partir de sa
composition (dosages en kg/m³) et de son âge (en jours).

Cas d'usage typiques :
- Aide à la formulation de mélanges béton en laboratoire
- Vérification préliminaire de résistance avant essais destructifs
- Analyse de l'influence des composants sur la résistance

### Usage hors périmètre
- Ne pas utiliser pour des décisions structurelles critiques sans validation
  expérimentale complémentaire (essais de compression réels)
- Ne pas extrapoler hors des plages de dosages du dataset (voir section 9)
- Non applicable à des types de béton spéciaux non représentés dans le dataset
  (béton fibré, béton projeté, etc.)

---

## 3. Facteurs

### Features d'entrée

| Feature | Unité | Min | Max | Moyenne |
|---|---|---|---|---|
| cement | kg/m³ | 102 | 540 | 281 |
| slag | kg/m³ | 0 | 359 | 74 |
| fly_ash | kg/m³ | 0 | 200 | 54 |
| water | kg/m³ | 122 | 247 | 182 |
| superplasticizer | kg/m³ | 0 | 32 | 6 |
| coarse_aggregate | kg/m³ | 801 | 1145 | 972 |
| fine_aggregate | kg/m³ | 594 | 993 | 774 |
| age | jours | 1 | 365 | 46 |

### Facteurs de variation identifiés
L'âge et le dosage en ciment sont les deux facteurs dominants (≈64% de
l'importance combinée). La relation entre ces variables et la résistance
est fortement non linéaire, ce qui justifie le choix d'un modèle à base
d'arbres.

---

## 4. Métriques

### Métrique principale
**RMSE (Root Mean Squared Error)** — en MPa. Choisie car elle exprime
l'erreur dans la même unité que la cible, ce qui la rend directement
interprétable par des ingénieurs.

### Métriques complémentaires
**Écart-type du RMSE sur les folds externes** — mesure la stabilité du
modèle. Un faible écart-type confirme que la performance n'est pas
le fruit d'un fold chanceux.

---

## 5. Données d'évaluation

**Dataset** : UCI Concrete Compressive Strength
([lien UCI](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength))

- **Taille** : 1 030 observations
- **Source** : I-Cheng Yeh, 1998
- **Preprocessing** : aucun (pas de valeurs manquantes, pas d'encodage
  nécessaire, toutes les variables sont numériques continues)
- **Méthode d'évaluation** : Nested Cross-Validation
  - Boucle externe : 5-fold KFold (shuffle=True, random_state=42)
  - Boucle interne : 3-fold KFold (shuffle=True, random_state=42)

---

## 6. Données d'entraînement

Identiques aux données d'évaluation — le dataset complet (1 030 obs.) est
utilisé pour le modèle final, après sélection des hyperparamètres par
GridSearchCV sur la boucle interne.

Aucune donnée externe n'a été utilisée.

---

## 7. Résultats quantitatifs

### Comparaison des 3 modèles (Nested CV, 5 folds externes)

| Modèle | RMSE moyen (MPa) | Écart-type | Rang |
|---|---|---|---|
| **Gradient Boosting** | **4.20** | **0.285** | **1** |
| Random Forest | 5.08 | 0.498 | 2 |
| Ridge | 10.51 | 0.809 | 3 |

### Interprétation
L'écart entre Ridge (10.51 MPa) et les modèles à base d'arbres confirme
la nature non linéaire de la relation entre composition et résistance.
Gradient Boosting surpasse Random Forest grâce au boosting séquentiel
des résidus, particulièrement efficace sur ce type de données tabulaires.

### Comparaison bonus — XGBoost
XGBoost testé en parallèle : RMSE = 4.253 ± 0.328 MPa. La différence
avec Gradient Boosting est négligeable (< 0.1 MPa) et ne justifie pas
la dépendance externe supplémentaire.

### Feature Importances (Gradient Boosting)

| Feature | Importance relative |
|---|---|
| age | 35% |
| cement | 29% |
| water | 11% |
| slag | 8.5% |
| superplasticizer | 8.3% |
| fine_aggregate | 4.5% |
| coarse_aggregate | 1.8% |
| fly_ash | 1.2% |

L'âge domine devant le ciment — physiquement logique : la réaction
d'hydratation continue longtemps après la coulée. Un béton vieux peut
surpasser un béton riche en ciment mais jeune.

---

## 8. Considérations éthiques

Ce modèle est développé dans un cadre académique sur des données
de laboratoire. Son usage dans un contexte de production réelle (chantier,
infrastructure critique) nécessite une validation rigoureuse par des
ingénieurs génie civil qualifiés.

Le dataset ne contient aucune donnée personnelle ou sensible.

---

## 9. Limites & Recommandations

**Limites identifiées :**
- Le modèle n'est fiable que dans les plages de dosages observées dans
  le dataset. Extrapoler au-delà (ex: ciment > 540 kg/m³) n'est pas garanti.
- Les bétons spéciaux (fibré, projeté, ultra-haute performance) ne sont
  pas représentés — les prédictions seraient non fiables.
- Dataset relativement petit (1 030 obs.) — les performances pourraient
  s'améliorer avec plus de données.

**Pistes d'amélioration :**
- Feature engineering : le ratio eau/ciment est un indicateur classique
  en génie civil, son ajout pourrait améliorer les performances.
- Tester LightGBM ou XGBoost avec un budget de tuning plus large.
- Collecter des données sur des types de béton plus variés pour élargir
  le domaine d'applicabilité.