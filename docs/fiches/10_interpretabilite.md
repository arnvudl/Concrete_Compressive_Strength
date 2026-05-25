# Fiche 10 — Interpretabilité & Specialized Focus

> Synthèse 1 — ML Basics (Prédiction vs Explication)

---

## Prédiction vs Explication (Synthèse 1)

| Objectif | Question posée | Modèle adapté |
|---|---|---|
| **Prédiction** | Quelle résistance pour cette formulation ? | N'importe lequel — même boîte noire |
| **Explication** | Pourquoi cette formulation est-elle résistante ? Quel ingrédient ajuster ? | Modèle interprétable |

**Dans la construction, on veut les DEUX.**

Un ingénieur civil ne veut pas juste "42 MPa". Il veut savoir : *"Si j'augmente le ciment de 50 kg/m³, est-ce que je compense la réduction d'eau ?"*

---

## Pourquoi l'interprétabilité est critique en génie civil

1. **Sécurité** : un modèle qu'un ingénieur ne peut pas expliquer à son client ou au bureau de contrôle n'est pas utilisable.
2. **Formulation** : on veut comprendre l'impact de chaque ingrédient pour optimiser la formule.
3. **Réglementation** : les normes EN 206 demandent des justifications, pas des boîtes noires.

---

## Pourquoi on a exclu les Neural Networks

| Modèle | Interprétabilité | Performance | Décision |
|---|---|---|---|
| Ridge | ✅ Très haute (coefficients directs) | ⚠️ Baseline | ✅ Inclus |
| Random Forest | ⚠️ Feature Importance | ✅ Bonne | ✅ Inclus |
| Gradient Boosting | ⚠️ Feature Importance | ✅ Meilleure | ✅ Inclus |
| **Neural Networks** | **❌ Boîte noire** | ✅✅ | **❌ Exclu** |

→ On a sacrifié potentiellement quelques points de RMSE pour maintenir l'interprétabilité.

---

## Feature Importance — Comment ça marche

### Impurity Importance (RF/GB)

**Cause** : mesure la somme des réductions de MSE obtenues quand la feature est utilisée pour un split, sur tous les arbres.

**Résultats GB** :

| Feature | Importance | Physique |
|---|---|---|
| `age` | ~35% | Hydratation log — relation non-linéaire capturée par les arbres |
| `cement` | ~29% | Liant principal — plus de ciment = plus de réactions |
| `water` | ~11% | Loi de Féret — eau en excès = pores = fragilité |
| `slag` | ~8.5% | Substitut liant lent |
| `superplasticizer` | ~8.3% | Réducteur d'eau indirect |
| `fine_agg` | ~4.5% | Remplissage |
| `coarse_agg` | ~1.8% | Remplissage |
| `fly_ash` | ~1.2% | Très faible concentration |

**age + cement = ~64% de l'importance combinée** — cohérent avec la physique.

### Limite de l'Impurity Importance

> Synthèse 7 : *"Favorise les features continues à haute cardinalité"*

Les features avec beaucoup de valeurs possibles (comme `cement`) peuvent être légèrement surévaluées par rapport à celles avec beaucoup de zéros (comme `fly_ash`).

Solution plus robuste : **Permutation Feature Importance** (permuter les valeurs d'une feature et mesurer la dégradation du score).

---

## Coefficients Ridge — L'interprétation la plus directe

```
cement  = +12.05  → le plus impactant positivement
slag    = +8.40
age     = +7.13
water   = -3.37   → le seul négatif (loi de Féret)
```

**Pourquoi comparables** : le StandardScaler ramène tout à moyenne=0, std=1 → les coefficients sont en "unités d'écart-type", donc directement comparables.

**Limite** : Ridge suppose des relations linéaires. `age` a une vraie relation logarithmique → le coefficient Ridge sous-estime l'impact réel de l'âge.

---

## Le lien Feature Importance ↔ Physique du béton

**`age` domine** → hydratation progressive (la réaction ciment + eau continue des semaines après la coulée)  
**`cement` fort** → plus de liant = plus de réactions = plus résistant  
**`water` négatif** → loi de Féret (1897) : résistance ∝ 1/(eau/ciment)²  
**`coarse_agg`, `fine_agg` faibles** → juste du remplissage, pas de contribution chimique

> *"Les Feature Importances confirment la physique du béton — ce n'est pas juste un modèle qui marche, c'est un modèle qui a appris des lois physiques réelles."*

---

## À retenir pour l'oral

> *"On a exclu les réseaux de neurones parce que l'interprétabilité est critique en génie civil — un ingénieur doit pouvoir expliquer ses choix de formulation à un bureau de contrôle. On a Ridge pour l'interprétabilité maximale (coefficients directs), et Feature Importance pour RF/GB. Le fait que `age` et `cement` dominent à 64% confirme que notre modèle a bien appris les lois physiques de l'hydratation et du dosage."*
