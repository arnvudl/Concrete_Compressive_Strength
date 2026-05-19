# 9 - Tuning

Date de création: 21 avril 2026 16:59
Cours: Machine Learning
Quadrimestre: Q2
État: Terminé

# **1. Fondations & Définitions du Problème**

---

Un modèle de ML sans tuning de ses hyperparamètres, c'est un sportif de haut niveau qui court avec des chaussures au hasard : le talent (l'algorithme) existe, mais la configuration (λ) détermine si la performance est optimale ou catastrophique.

![image.png](image.png)

---

## Paramètres θ vs Hyperparamètres λ

Les **paramètres du modèle** θ sont appris automatiquement par le processus d'entraînement via la minimisation du risque empirique :

$$
\hat{\theta} = \arg\min_{\theta} \mathcal{R}_{emp}(\theta)
$$

---

Les **hyperparamètres** λ sont des entrées du processus d'apprentissage lui-même. Ils ne sont **pas** optimisés pendant l'entraînement : ils doivent être fixés avant. Ils contrôlent la complexité, la structure et les décisions computationnelles du modèle.

**Types de hyperparamètres :**

| Type | Exemples |
| --- | --- |
| **Réel-continu** | Min. amélioration d'erreur pour split dans un arbre ; bandwidth kernel Naive Bayes |
| **Entier** | Taille du voisinage k dans k-NN ; `mtry` dans Random Forest |
| **Catégoriel** | Critère de split (Gini vs Entropie) ; mesure de distance dans k-NN |
| **Hiérarchique/conditionnel** | Si kernel = density estimate → sa largeur s'active en tant que HP |

⚠️ Les HPs sont souvent **dépendants les uns des autres** (structure hiérarchique), ce qui complexifie massivement l'espace de recherche.

---

<aside>
💡

Avant de lancer un `RandomForestClassifier`, `n_estimators`, `max_depth`, `min_samples_split` sont tous des λ. `sklearn` ne les optimise pas pour vous, c'est votre job.

</aside>

---

## Erreur de Généralisation & Objectif HPO

L'**erreur de généralisation estimée** mesure la performance du modèle $\hat{f} = I(\mathcal{D}_{train}, \lambda)$ sur des données non vues :

$$
GE^d_{\mathcal{D}{train}, \mathcal{D}{test}}(I, \lambda, n_{train}, \rho) = \rho\left(y_{\mathcal{D}{test}},\ F{\mathcal{D}_{test}, \hat{f}}\right)
$$

où $ρ$ est la mesure de performance (AUC, accuracy, MSE...), $F_{\mathcal{D}_{test}, \hat{f}}$ les prédictions sur le test set.

**Problème formel HPO :**

$$
\lambda^* \in \arg\min_{\lambda \in \tilde{\Lambda}}\ c(\lambda) = \arg\min_{\lambda \in \tilde{\Lambda}}\ GE^d(I, \mathcal{J}, \rho, \lambda)
$$

où $\tilde{\Lambda} = \tilde{\Lambda}_1 \times \tilde{\Lambda}_2 \times \cdots \times \tilde{\Lambda}_l$ est l'espace de recherche (sous-ensemble borné de $Λ$).

**Le tuner comme fonction formelle :**

$$
\tau : (\mathcal{D},\ I,\ \tilde{\Lambda},\ \mathcal{J},\ \rho)\ \mapsto\ \hat{\lambda}
$$

L'archive des évaluations s'accumule itérativement :

$$
\mathcal{A}^{[t+1]} = \mathcal{A}^{[t]} \cup \left(\lambda^+,\ c(\lambda^+)\right)
$$

---

## Pourquoi le Tuning est Difficile

Quatre propriétés rendent le HPO fondamentalement dur :

| Propriété | Explication | Conséquence technique |
| --- | --- | --- |
| **Boîte noire** | Vous ne pouvez regarder à l'intérieur : vous mettez un $λ$, vous recevez un score, c'est tout | Pas de gradient $∇c(λ)$ disponible |
| **Coût élevé** | Chaque test demande d'entraîner le modèle entier | Budgets d'évaluation limités (dizaines, pas millions) |
| **Stochasticité** | Le score que vous recevez n'est pas exact, il varie selon la partition du resampling | Évaluations bruitées, pas déterministes |
| **Structure complexe** | Mélange de types + dépendances hiérarchiques | Pas d'espace métrique standard, algorithmes classiques inadaptés |

💡 Conséquence directe : croissance **exponentielle** de l'espace $Λ$. Deux HPs discrets de 10 valeurs chacun = 100 configurations. Cinq HPs = 100 000. La force brute devient impossible.

---

# Techniques de Base : Grid Search & Random Search

**Grid Search** et **Random Search** sont les deux méthodes de référence ("naïves") pour explorer $Λ̃$ : l'une couvre systématiquement, l'autre couvre intelligemment — comprendre leur différence structurelle conditionne tous les choix algorithmiques avancés.

---

## Grid Search

Grid Search énumère **toutes les combinaisons possibles** d'un ensemble fini de valeurs candidates prédéfinies pour chaque hyperparamètre. Pour $l$ hyperparamètres avec $n_i$ valeurs chacun, le nombre total d'évaluations est :

$$
|\tilde{\Lambda}{grid}| = \prod{i=1}^{l} n_i
$$

Exemple : 3 HPs × 10 valeurs chacun = $10^3 = 1000$ évaluations.

**Mécanique étape par étape :**

1. Définir manuellement une grille discrète pour chaque HP : ${v_1^{(i)}, v_2^{(i)}, \ldots, v_{n_i}^{(i)}}$
2. Construire le produit cartésien : $\tilde{\Lambda}_{grid} = {v^{(1)}} \times {v^{(2)}} \times \cdots \times {v^{(l)}}$
3. Évaluer $c(\lambda)$ pour chaque $\lambda \in \tilde{\Lambda}_{grid}$ (ordre arbitraire, parallélisable)
4. Retourner $\hat{\lambda} = \arg\min_{\lambda \in \tilde{\Lambda}_{grid}} c(\lambda)$

---

### Analogie : Minecraft

Grid Search, c'est miner le sol bloc par bloc sur une grille parfaite 10×10. Vous êtes certain de ne rater aucune case. Mais si le diamant est entre deux blocs, à une profondeur que vous n’avez pas checkée, vous passez à côté pour toujours.

---

## Random Search

Random Search **échantillonne uniformément** depuis l'espace de recherche continu $\tilde{\Lambda}$ sans discrétisation imposée. Pour un budget de $B$ évaluations :

$$
\lambda^{(b)} \sim \mathcal{U}(\tilde{\Lambda}), \quad b = 1, \ldots, B
$$

**Mécanique étape par étape :**

1. Définir les bornes continues (ou distributions) de chaque HP
2. Tirer $B$ configurations iid depuis $\mathcal{U}(\tilde{\Lambda})$
3. Évaluer $c(\lambda^{(b)})$ pour chaque tirage
4. Retourner $\hat{\lambda} = \arg\min_{b} c(\lambda^{(b)})$

C'est un **algorithme anytime** : on peut stopper à tout moment avec le meilleur résultat obtenu jusqu'alors, ou continuer indéfiniment.

---

### Analogie : Algo TikTok

L'algo TikTok ne scanne pas chaque créateur de façon systématique : il teste des combinaisons aléatoires de profils et mesure l'engagement. Parfois il tombe sur une pépite que le scan systématique aurait mis 10x plus de temps à trouver, parce qu'il couvre l'espace de manière non uniforme dans les dimensions qui comptent.

---

## Pourquoi Random Search domine Grid Search

Le résultat clé est illustré par ce cas :

**Fonction objectif :** $f(x_1, x_2) = g(x_1) + h(x_2) \approx g(x_1)$

→ Seul $x_1$ influence vraiment la performance. $x_2$ est quasi-inutile.

Avec Grid Search 5×5 = 25 évaluations : **seulement 5 valeurs distinctes** de $x_1$ sont testées (les 5 lignes de la grille).

Avec Random Search 25 évaluations : **25 valeurs distinctes** de $x_1$ sont testées (chaque point est unique sur les deux axes).

<aside>
⚠️

**La leçon fondamentale :** Grid Search gaspille son budget à couvrir les dimensions inutiles avec une résolution fixe. Random Search alloue naturellement plus de couverture aux dimensions qui comptent, sans savoir lesquelles a priori.

</aside>

---

Voici le diagramme comparatif du flux de décision des deux algorithmes :

![image.png](image%201.png)

---

## Tableau de Confrontation

| Critère | Grid Search | Random Search |
| --- | --- | --- |
| **Espace couvert** | Discrétisation fixe imposée | Continu, sans discrétisation |
| **Scalabilité** | ❌ Exponentielle en $l$ | ⚠️ Mauvaise en haute dimension |
| **Parallelisation** | ✅ Triviale | ✅ Triviale |
| **Arrêt anticipé** | ❌ Doit finir la grille entière | ✅ Anytime — stop quand voulu |
| **HP non pertinents** | ❌ Gaspille des evals dessus | ✅ Couverture naturellement meilleure |
| **Implémentation** | ✅ Triviale | ✅ Triviale |
| **Types de HP** | ✅ Tous (réel, entier, catég.) | ✅ Tous |
| **Efficacité dans zones prometteuses** | ❌ Aucune exploitation | ❌ Aucune exploitation |
| **Référence pratique** | `GridSearchCV` sklearn | `RandomizedSearchCV` sklearn |

### **Diagnostic métier — Qui en souffre / Qui l'évite :**

<aside>
💀

**Souffre de Grid Search :** Tout projet avec >3 HPs continus. Un SVM avec `C`, `gamma`, `epsilon` sur 10 valeurs chacun = 1 000 entraînements complets. Avec un modèle qui prend 10 min à entraîner = 7 jours de calcul.

</aside>

<aside>
✅

**Évite avec Random Search :** Même budget de 100 évaluations, couverture réelle de 100 valeurs distinctes par HP. Si seul `C` compte vraiment, Random Search le détecte avec 20x moins de gâchis.

</aside>

<aside>
⚠️

**Les deux souffrent :** Espaces de haute dimension (>10 HPs), où même 10 000 évaluations aléatoires ne couvrent qu'une infime fraction de Λ̃. → Nécessite les techniques avancées (Section 3).

</aside>

---

## Résultat Empirique : Random Forest sur données Sonar

Expérience documentée dans le cours : tuning d'un Random Forest avec **5-fold CV** sur le dataset `sonar`, métrique AUC.

**HPs optimisés :**

| HP | Type | Min | Max |
| --- | --- | --- | --- |
| `num.trees` | Integer | 3 | 500 |
| `mtry` | Integer | 5 | 50 |
| `min.node.size` | Integer | 10 | 100 |

**Résultat :** Random Search atteint AUC ≈ 0.940 dès ~50 itérations. Grid Search atteint AUC ≈ 0.934 après 220 itérations. Random Search converge plus vite vers un meilleur optimum sur cet espace 3D.

---

![image.png](image%202.png)

---

# Techniques Avancées de Tuning

Grid Search et Random Search ignorent totalement ce qu'ils ont déjà évalué — les techniques avancées exploitent l'historique des évaluations pour **apprendre où chercher** (BO) ou **tuer les mauvais candidats tôt** (SH/Hyperband), réduisant le budget d'un facteur 10x à 100x.

---

## Algorithmes Évolutionnaires

Méthodes d'optimisation stochastiques **basées sur une population**, inspirées de l'évolution biologique. Applicables au HPO car elles **ne nécessitent pas de gradient** de $c(\lambda)$.

### **Deux opérateurs fondamentaux :**

- **Mutation :** Changement aléatoire d'un ou plusieurs HPs dans une configuration
    
    $$
    \lambda^{mut} = \lambda + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
    $$
    
- **Crossover** : création d'une nouvelle HPC en mixant deux configurations parentes
    
    $$
    \lambda^{off}_i = \begin{cases} \lambda^{(1)}_i & \text{avec proba } p \ \lambda^{(2)}_i & \text{avec proba } 1-p \end{cases}
    $$
    

---

### **Mécanique étape par étape**

1. **Initialisation** : générer une population ${\lambda^{(1)}, \ldots, \lambda^{(pop)}}$ aléatoirement, évaluer $c(\lambda^{(k)})$ pour chacun
2. **Sélection parentale** : choisir des parents selon un critère (uniforme, tournoi, roulette proportionnelle au fitness)
3. **Crossover** : combiner deux parents pour créer des "offspring" $\lambda^{(pop+1)}, \ldots, \lambda^{(pop+off)}$
4. **Mutation** : perturber aléatoirement chaque offspring
5. **Évaluation** : calculer $c(\lambda)$ pour les nouveaux individus
6. **Sélection de survie** : garder les $p^{(pop)}$ meilleurs (élitisme ou roulette)
7. **Critère d'arrêt** → si non atteint, retourner à l'étape 2

---

## **Analogie : Gacha Game**

Dans Genshin Impact, les meilleurs builds de personnages ("configurations $λ$") émergent de la communauté par "évolution" : les joueurs copient les builds performants (sélection), les modifient légèrement (mutation), combinent des éléments de deux builds (crossover). Les builds nuls disparaissent du méta (sélection de survie). Aucun joueur n'a calculé le build optimal analytiquement — il a émergé par pression de sélection.

---

## **Caractérisation HPO**

Les méthodes HPO se différencient par deux axes :

- **Exploration vs exploitation** : comment équilibrer zones inconnues vs zones prometteuses
- **Inférence vs recherche** : combien de compute dédier à apprendre le paysage vs l'explorer

---

![image.png](image%203.png)

---

## Optimisation Bayésienne (BO)

BO construit un **modèle probabiliste** (surrogate) de $c(\lambda)$ à partir des évaluations passées, puis optimise une **fonction d'acquisition** bon marché pour choisir le prochain candidat. C'est l'approche "apprendre le paysage + l'exploiter" par opposition à la "recherche aveugle".

---

### **Deux composantes formelles**

- **Modèle surrogate** : Régression non-linéaire probabiliste de $\lambda \mapsto c(\lambda)$
    
    $$
    C(\lambda) \sim \left(\hat{c}(\lambda),\ \hat{\sigma}(\lambda)\right)
    $$
    
    où $\hat{c}(\lambda)$ est la moyenne postérieure (prédiction) et $\hat{\sigma}(\lambda)$ l'incertitude. Choix typiques : **Processus Gaussiens** (espaces numériques continus), **Forêts aléatoires** (espaces mixtes avec HP catégoriels).
    
- **Fonctions d'acquisition** : Choisir le prochain $λ$  à évaluer en équilibrant exploration (haute $\hat{\sigma}$) vs exploitation (faible $\hat{c}$) :
    
    $$
    \text{LCB :}\quad a(\lambda) = \hat{c}(\lambda) - \kappa \cdot \hat{\sigma}(\lambda)
    $$
    
    $$
    EI :a(λ)=E[max{c_{min}−C(λ), 0}]
    $$
    
    où $c_{min}$ est la meilleure valeur observée dans l'archive $\mathcal{A}$.
    

---

### **Mécanique BO - boucle séquentielle :**

1. Évaluer un **design initial** de quelques HPCs aléatoires → archive $\mathcal{A}$
2. Entraîner le surrogate $\hat{c}(\lambda)$ sur $\mathcal{A}$
3. Optimiser $a(\lambda)$ sur $Λ̃$ → proposer $\lambda^+ = \arg\max_\lambda a(\lambda)$
4. Évaluer $c(\lambda^+)$ par resampling → ajouter à $\mathcal{A}$
5. Retour à l'étape 2 jusqu'à budget épuisé

<aside>
⚠️

Optimiser $a(\lambda)$ reste difficile mais **beaucoup moins coûteux** que d'évaluer $c(\lambda)$ (pas d'entraînement ML complet requis).

</aside>

---

### **Analogie : Algorithme de recommandation Spotify**

Spotify ne joue pas toutes les chansons du monde pour trouver vos préférées. Il construit un **modèle de vos goûts** (surrogate) à partir de ce que vous avez écouté, puis vous recommande des chansons qui maximisent une fonction d'acquisition : ni trop semblables (exploitation pure) ni trop différentes (exploration pure). Chaque écoute met à jour le modèle. C'est exactement BO appliqué à l'espace des HPs.

---

![image.png](image%204.png)

---

## Optimisation Multi-Fidelité

**Prérequis : le HP de fidelité $\lambda_{fid}$**

$\lambda_{fid}$ est une composante de $λ$ qui contrôle le **coût computationnel** de façon monotone croissante : plus $\lambda_{fid}$ est élevé, plus l'évaluation est coûteuse mais précise.

Exemples concrets :

- Nombre d'**epochs SGD** pour un réseau de neurones
- **Taille du training set** utilisé pour l'entraînement
- Nombre de **rounds de boosting**

Formellement : $\lambda_{fid} \in [\lambda_{fid}^{low},\ \lambda_{fid}^{upp}]$ où $\lambda_{fid}^{upp}$ correspond à la fidelité maximale (coût maximal, résultat le plus proche du vrai $c(\lambda)$).

**Insight fondamental :** Baisser $\lambda_{fid}$ permet d'explorer plus de points dans $Λ̃$, mais avec une information moins fiable sur leur vraie performance.

---

### a. Successive Halving (SH)

**Définition :** SH démarre avec $p^{[0]}$ configurations, les entraîne avec un **petit budget** $\lambda_{fid}^{[0]}$, élimine les $\frac{\eta-1}{\eta}$ pires, multiplie le budget par $\eta$ pour les survivants. Répète jusqu'à un seul survivant ou budget épuisé.

**Règle de mise à jour à chaque round $t$ :**

$$
p^{[t+1]} = \left\lfloor \frac{p^{[t]}}{\eta} \right\rfloor, \qquad \lambda_{fid}^{[t+1]} = \eta \cdot \lambda_{fid}^{[t]}
$$

**Exemple avec $η=2$, départ $p=8$, budget=1 :**

| Round $t$ | Budget $\lambda_{fid}^{[t]}$ | Configs $p^{[t]}$ |
| --- | --- | --- |
| 0 | 1 (12%) | 8 |
| 1 | 2 (25%) | 4 |
| 2 | 4 (50%) | 2 |
| 3 | 8 (100%) | 1 |

<aside>
⚠️

**Problème de SH :** une bonne configuration peut être éliminée au round 0 si elle est lente à démarrer (mauvaise performance à faible budget) mais excellente à budget complet. C'est le **problème du "promising-but-slow learner"**.

</aside>

<aside>
💡

**Analogie X Factor / La Voix :** Le jury élimine les candidats après 30 secondes de chanson. Un candidat dont la voix chauffe après 2 minutes est éliminé prématurément. SH souffre du même biais.

</aside>

---

### b. Hyperband

Hyperband répète SH avec **différents points de départ** $(p^{[0]}, \lambda_{fid}^{[0]})$. Chaque run SH = un **bracket**. Chaque bracket consomme approximativement le **même budget total**.

**Structure pour $η=4$ :**

| Bracket | $\lambda_{fid}^{[0]}$ | $p^{[0]}$ | Rounds | Logique |
| --- | --- | --- | --- | --- |
| 3 (large) | 1 | 82 | 4 | Beaucoup de configs, petits budgets |
| 2 | 4 | 27 | 3 | Intermédiaire |
| 1 | 16 | 10 | 2 | Intermédiaire |
| 0 (pur) | 64 | 5 | 1 | Peu de configs, gros budget d'emblée |

<aside>
💡

Le bracket 0 est un Random Search avec budget complet : il protège contre le cas où toutes les bonnes configs sont "slow starters". Le bracket 3 maximise l'exploration à faible coût.

</aside>

---

## Tableau de Confrontation Global : Les 5 Méthodes HPO

| Critère | Grid | Random | Evolutionnaire | BO | Hyperband |
| --- | --- | --- | --- | --- | --- |
| **Exploite l'historique** | ❌ | ❌ | ⚠️ partiel | ✅ central | ❌ |
| **Réduit les evals coûteuses** | ❌ | ❌ | ❌ | ❌ | ✅ central |
| **Espaces haute dim.** | ❌ | ⚠️ | ✅ | ⚠️ GP scale mal | ✅ |
| **HP catégoriels/conditionnels** | ✅ | ✅ | ✅ | ⚠️ GP non, RF oui | ✅ |
| **Parallélisable** | ✅ | ✅ | ✅ | ⚠️ séquentiel natif | ✅ |
| **Coût impl./setup** | faible | faible | moyen | élevé | moyen |
| **Référence pratique** | sklearn | sklearn | DEAP, CMAES | Optuna, SMAC3 | Ray Tune, Optuna |

**Diagnostic métier :**

✅ **BO pour :** Budget < 100 évaluations, espace continu bas-dimensionnel (<10 HPs), chaque entraînement coûte cher (>1h). → Chaque évaluation doit compter.

✅ **Hyperband/SH pour :** Deep learning avec epochs contrôlables, large espace, budget compute fixe. → Éliminer vite les configs inutiles.

✅ **Evolutionnaire pour :** Espaces très larges, non-différentiables, problèmes multi-objectifs. → Robustesse sur structures complexes.

❌ **Grid Search en 2025 :** Uniquement pour 1-2 HPs catégoriels avec peu de valeurs. Sinon toujours suboptimal.

---

# Évaluation Fiable, Nested Resampling & Pipelines/AutoML

Trouver $λ*$ est inutile si le score qu'on annonce au client est biaisé et intégrer le preprocessing dans le tuning sans précaution revient à tricher à l'examen en regardant les réponses avant de commencer.

---

## Le Principe du Test Set Intouché

**Le piège classique :** Après avoir trouvé $λ*$ par tuning sur (Train + Validation), on évalue la performance finale **sur les mêmes données**. Le résultat est **optimistement biaisé** : le tuner a indirectement "vu" le test set via les multiples évaluations qui l'ont guidé.

**Pourquoi c'est un biais :** Sélectionner le meilleur λ parmi 100 candidats évalués sur le même jeu revient à choisir le meilleur résultat de 100 tests — par chance pure, ce résultat sera toujours trop optimiste.

**La règle:** Le test set doit rester **totalement intouché** pendant toute la phase de tuning. Il sert uniquement à l'évaluation finale du modèle entraîné avec $λ*$.

### **Mécanique du découpage en 3 parties :**

- **Train** : entraîner le modèle avec chaque λ candidat
- **Validation** : mesurer c(λ) pour guider le tuner → optimiser λ
- **Test** : mesurer la performance finale du modèle entraîné avec λ* sur Train+Validation

![image.png](image%205.png)

---

## Nested Resampling

**Problème du split simple :** Un seul holdout (Train/Valid/Test) donne une estimation de GE à **haute variance** — elle dépend fortement du tirage aléatoire de la partition. Sur un petit dataset, un mauvais split peut fausser tout le diagnostic.

**Solution → Nested Resampling :** On généralise le split 3 parties en deux boucles de resampling imbriquées.

---

**Boucle externe** (k-fold extérieur) — estime la performance du **processus de tuning complet** :

- À chaque fold externe, une portion des données est mise de côté comme test externe
- Sur le reste (données d'entraînement externe), on lance la boucle interne

**Boucle interne** (k-fold intérieur) — réalise le **tuning** proprement dit :

- Sur les données d'entraînement externe, on divise à nouveau en train/validation interne
- On évalue tous les λ candidats, on sélectionne λ*
- Ce λ* remonte à la boucle externe pour être évalué sur le test externe

> **Ce que ça garantit :** Le test externe n'a jamais été vu pendant le tuning interne. La GE estimée est donc **non biaisée** et à **variance réduite** (car moyennée sur plusieurs folds externes).
> 

---

<aside>
⚠️

**Coût computationnel :** Avec k-fold externe × k-fold interne, on multiplie les entraînements. Exemple : 5 folds externes × 3 folds internes × 50 configs HPO = **750 entraînements complets**. C'est pourquoi les techniques multi-fidelité (SH/Hyperband) s'intègrent naturellement dans la boucle interne.

</aside>

---

## Pipelines ML

Un pipeline est une séquence de nœuds apprenants connectés où les données s'écoulent de source ($D_{train}$) vers sortie (prédictions). Chaque nœud possède une **phase train** et une **phase predict**, et apprend ses propres paramètres.

### **Pourquoi les pipelines sont obligatoires avec le tuning ?**

Si on normalise les données **avant** de créer les folds CV, la normalisation a utilisé des statistiques calculées sur tout D (incluant le fold de validation). C'est une **fuite de données** (data leakage) — le modèle "connaît" indirectement la distribution du validation set. Le pipeline encapsule le preprocessing **à l'intérieur** de la CV, garantissant que chaque fold n'est transformé qu'avec les statistiques du train de ce fold.

#### **Pipeline séquentiel**

Les nœuds sont en série — Scaling → Encoding → Feature Selection → Learner. Chaque nœud reçoit la sortie du précédent.

#### **Pipeline DAG (Directed Acyclic Graph)**

- Une **source unique** accepte D_train
- Un **sink unique** retourne les prédictions
- Les nœuds intermédiaires peuvent être des opérations de preprocessing, des learners, des opérations de postprocessing, ou des **contrôleurs de flux** (branchement conditionnel)
- Permet d'implémenter des ensembles, de la sélection d'opérateur, du feature engineering conditionnel

**L'espace de HPs d'un pipeline** est le produit cartésien des espaces de tous ses nœuds. Dans un DAG avec branchement, certains nœuds et leurs HPs sont **actifs ou inactifs** selon les décisions de branchement → espace hiérarchique conditionnel.

![image.png](image%206.png)

---

## AutoML

AutoML combine un **pipeline DAG** complet (preprocessing + sélection de learner + HPs) avec un **tuner efficace** pour automatiser l'ensemble du workflow ML. 
Le problème est souvent appelé **CASH** (Combined Algorithm Selection and Hyperparameter optimization).

**Le cas pour AutoML :**

- De plus en plus de tâches abordées par des méthodes data-driven
- Les data scientists s'appuient sur le trial-and-error — coûteux et non reproductible
- Les tâches similaires et récurrentes gagnent à être automatisées

**Ce qu'AutoML automatise :**

- Sélection du type de preprocessing (scaling, encoding, imputation)
- Sélection de l'algorithme (SVM, Random Forest, Gradient Boosting...)
- Tuning conjoint de tous leurs HPs dans un espace joint Λ̃ hiérarchique

**Ce qu'AutoML ne peut pas (encore) automatiser :**

- La définition de l'objectif métier
- La collecte et validation des données
- L'interprétabilité et l'audit du modèle
- Le déploiement et la surveillance en production

**Défis ouverts identifiés dans le cours :**

- Quelle est l'approche la plus efficace pour explorer l'espace CASH ?
- Comment intégrer la connaissance a priori humaine (warmstarts, learned search spaces) ?
- Gestion des objectifs multiples incluant l'interprétabilité du modèle
- AutoML reste une boîte noire — frein à l'adoption dans des contextes réglementés

---

**Diagnostic métier — Qui l'utilise :**

✅ **AutoML pertinent :** Tâches tabular bien définies, datasets propres, contrainte de temps forte, équipes sans data scientists seniors. Outils : `auto-sklearn`, `FLAML`, `AutoGluon`, `H2O AutoML`.

❌ **AutoML insuffisant :** Computer vision / NLP avec architectures custom, données mal labelisées, exigences d'explicabilité légale (RGPD, secteur bancaire/médical), when the pipeline structure itself is the research question.

---