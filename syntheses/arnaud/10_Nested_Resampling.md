# 10 - Nested Resampling

Date de création: 28 avril 2026 21:57
Cours: Machine Learning
Quadrimestre: Q2
État: Terminé

# A. RÉCAP du chapitre précédent

Trouver $λ*$ ne suffit pas; si le score annoncé est biaisé par le processus de tuning lui-même, le modèle "performant" en développement sera décevant en production.

---

## A1. Problème → Overtuning

---

### **Setup de l'expérience révélatrice**

Classifieur binaire **feature-indépendant :** λ n'a **aucun effet réel** sur les prédictions. Le modèle tire des labels aléatoirement. Sa vraie GE = **50% exactement**.

On "tune" quand même : 100 valeurs de λ testées, expérience répétée 50 fois.

---

### **Ce qui se passe**

Pour 1 run de CV avec λ fixé, le score tourne autour de 0.5 → correct, comme attendu. Les scores suivent une distribution binomiale recentrée autour de 0.5.

**Mais** en tuning, on prend le **minimum** des scores sur tous les λ testés ; pas la moyenne. On n'estime plus la performance moyenne, on estime la **performance du meilleur cas par chance pure**.

---

### **Les deux effets amplificateurs du biais**

| Facteur | Effet sur le biais |
| --- | --- |
| ↑ Nombre de configs testées | ↑ Biais (plus de chances de tomber sur un faux minimum) |
| ↓ Taille du dataset | ↑ Biais (moins de données = plus de variance par fold = plus facile de "tomber bien") |

---

### **Résultat empirique**

Avec n=100 observations et 100 configs testées, le "meilleur" score CV descend artificiellement vers ~0.38 alors que la vraie GE est 0.5. Le modèle semble meilleur de 12 points — sur un classifieur aléatoire.

<aside>
⚠️

Le cours établit un parallèle explicite avec le **problème des tests multiples** en statistique : plus on teste d'hypothèses, plus la probabilité d'obtenir un faux positif significatif augmente. C'est exactement la même mécanique.

</aside>

---

### **Analogie : loot box**

Tu ouvres 100 loot boxes dans un jeu. Par chance pure, une te donne un item "légendaire" avec 5% de stats en plus. Si tu annonces "mon perso a +5% de stats", tu mens — ce résultat est le fruit du cherry-picking sur 100 tirages, pas de la vraie puissance du build. Le tuner sans untouched test set fait exactement ça.

---

![image.png](image.png)

---

## A2. La Solution → Untouched Test Set

---

### **Règle absolue**

Toutes les étapes du model building — sélection de modèle, preprocessing, tuning — doivent être réalisées **exclusivement sur les données d'entraînement**. Le test set n'est touché **qu'une seule fois**, après que λ* est définitivement fixé et le modèle final entraîné.

---

### **Split 3 parties : protocole exact**

Pendant le tuning : le learner s'entraîne sur **Train**, est évalué sur **Validation** → le tuner optimise λ sur cette boucle.

Après sélection de $λ*$ : re-entraînement sur **Train + Validation** joint → évaluation finale sur **Test** (intouché).

<aside>
💡

**Insight clé — le tuning comme partie de l'entraînement :** 
On peut réinterpréter le processus entier comme un algorithme "self-tuning". Les HPs disparaissent des inputs visibles de l'algorithme — ils sont résolus en interne. Le dataset Train+Validation devient le dataset d'entraînement de cet algorithme augmenté. Le test set évalue ce système complet, pas juste le modèle final.

</aside>

---

## A3. La Généralisation : Nested Resampling

---

### **Pourquoi généraliser le split 3-voies**

Un seul holdout donne une estimation à haute variance ; elle dépend du tirage aléatoire de la partition. La généralisation naturelle est le nested resampling : on remplace chaque split unique par une boucle de CV.

---

### **Procédure exacte : exemple 3-fold externe × 4-fold interne**

**Pour chaque fold de la boucle externe :**

1. Isoler le **test externe** (vert clair) : il ne sera plus touché dans cette itération
2. Sur le **train externe** (vert foncé), lancer le tuner via **4-fold CV interne** (bleu/gris), évaluer chaque $λᵢ$
3. Retourner le $λ*$ qui performe le mieux sur les **test sets internes** (gris)
4. Re-entraîner le modèle sur le **train externe complet** avec ce $λ*$
5. Évaluer sur le **test externe** (vert clair) → score non biaisé pour ce fold

**Agréger les scores** des 3 folds externes → estimation finale de GE robuste et non biaisée.

**Garantie d'absence de biais :** Le test externe n'a jamais participé au tuning interne. Le modèle évalué sur lui n'a "vu" ces données à aucun moment de son processus de construction.

---

![image.png](image%201.png)

---

## A4. La Preuve Empirique Conclusive

---

### **Retour sur l'expérience du classifieur aléatoire, avec nested resampling ajouté**

**Sans nested resampling (pointillés) :** la performance "tuned" plonge artificiellement vers 0.30-0.45 selon la taille du dataset, et continue de se dégrader avec plus de configs testées.

**Avec nested resampling (lignes pleines) :** la performance estimée reste **stable autour de 0.50** — la vraie GE — quelle que soit la quantité de configs testées et quelle que soit la dimension des données.

**L'effet taille de données :** Plus $n$ est petit ($n=50$, courbe violette), plus la variance de l'estimation est élevée — les oscillations sont plus fortes. Mais la **moyenne reste centrée sur 0.50**. Le biais est corrigé, seule la variance résiduelle dépend de n.

**Conséquence directe :** Le nested resampling ne rend pas le tuning inutile — il rend son **évaluation honnête**. Un bon λ reste bon, un mauvais dataset reste mauvais. Il enlève l'illusion.

---

# B. Nested Resampling

### **La chaîne causale complète du chapitre**

Le tuning sélectionne le **minimum** parmi plusieurs évaluations → ce minimum est biaisé vers le bas → plus on tune, plus le biais est grand → plus le dataset est petit, plus le biais est grand.

---

### **Countermeasure 1 : Split 3 voies**

Séparer physiquement les données de tuning (Train+Valid) des données d'évaluation finale (Test). Le test n'est touché qu'une seule fois, après décision définitive sur λ*.

---

### **Countermeasure 2 : Nested Resampling**

Généraliser le split 3 voies en deux boucles CV imbriquées pour réduire la variance de l'estimation. La boucle interne tune, la boucle externe évalue. Les deux sont strictement séparées.

---

### **Reinterpretation architecturale**

Le processus `[Run CV interne → sélectionner λ* → re-entraîner sur D]` constitue un **algorithme augmenté self-tuning**. Ses inputs sont les données brutes et l'inducer. Son output est un modèle. Les HPs ont disparu de l'interface visible. Le nested resampling évalue cet algorithme complet, pas juste son output final.

---

## **C. Tableau de confrontation : Simple CV vs Nested Resampling**

| Critère | CV simple (sans nested) | Nested Resampling |
| --- | --- | --- |
| **Biais de l'estimation GE** | ❌ Optimiste, croît avec #configs | ✅ Non biaisé |
| **Variance de l'estimation** | ⚠️ Modérée | ✅ Réduite (moyennée sur folds externes) |
| **Coût computationnel** | Faible | ❌ k_outer × k_inner × #configs |
| **Utilisation données** | Efficace | ⚠️ Chaque fold utilise moins de train |
| **Applicable à petits datasets** | ❌ Biais fort | ✅ Conçu pour ça |
| **Détecte l'overtuning** | ❌ Invisible | ✅ Visible : GE reste stable |