# Fiche 9 — Model Card (Mitchell et al. 2019)

> Framework : https://arxiv.org/abs/1810.03993

---

## C'est quoi une Model Card ?

Un document standardisé qui **documente un modèle ML** de manière transparente et honnête, pour que quelqu'un qui ne l'a pas construit puisse comprendre :
- Ce que le modèle fait (et ne fait pas)
- Comment il a été évalué
- Ses limites et risques

Créé par Mitchell et al. (2019) chez Google.

---

## Les 9 sections — Ce que chaque section veut dire

### 1. Model Details
**Quoi** : informations factuelles de base.  
**Contient** : algo, version, date, auteurs, HPs retenus, framework, protocole d'évaluation.  
**Notre réponse** : Gradient Boosting sklearn, `learning_rate=0.2, max_depth=4, n_estimators=300, subsample=1.0`, nested 5-fold CV.

### 2. Intended Use
**Quoi** : pour quoi le modèle est fait, et pour quoi il **ne l'est pas**.  
**Pourquoi important** : éviter les usages inappropriés (ex: prendre une décision structurelle critique basée uniquement sur ce modèle).  
**Notre réponse** :
- ✅ Aide à la formulation, analyse de sensibilité, recherche académique
- ❌ Validation réglementaire, bétons spéciaux, formulations hors plage

### 3. Factors
**Quoi** : variables qui font varier les performances du modèle.  
**Pourquoi important** : identifier où le modèle est moins fiable.  
**Notre réponse** : age rare (peu de données hors 28j), formulations à forte teneur en adjuvants (slag, fly_ash, SP souvent = 0), béton très résistant (> 70 MPa = peu d'observations).

### 4. Metrics
**Quoi** : comment on mesure la performance, pourquoi ces métriques.  
**Notre réponse** : RMSE en MPa (principale — lisible par un ingénieur), R² (complémentaire). La métrique interne GridSearchCV = neg_MSE.

### 5. Evaluation Data
**Quoi** : quel dataset, comment préparé, quel protocole.  
**Notre réponse** : UCI Concrete (Yeh, 1998), 1 005 observations après nettoyage, nested 5-fold CV avec KFold(shuffle=True, random_state=0 outer / 42 inner), zéro leakage via Pipeline.

### 6. Training Data
**Quoi** : données utilisées pour le modèle final.  
**Notre réponse** : même dataset — modèle final entraîné sur 1 005 observations avec HPs optimaux.

### 7. Quantitative Analyses
**Quoi** : les vrais chiffres, par modèle et par sous-groupe.  
**Notre réponse** :

| Modèle | RMSE | R² |
|---|---|---|
| Ridge | 10.385 ± 0.449 MPa | 0.590 |
| RF | 4.935 ± 0.318 MPa | 0.907 |
| **GB** | **4.208 ± 0.261 MPa** | **0.932** |

Feature Importances GB : age (35%), cement (29%), water (11%).

### 8. Ethical Considerations
**Quoi** : risques, données sensibles, usages dangereux.  
**Notre réponse** : aucune donnée sensible (mesures physiques). Risque principal = décision structurelle sans validation physique → toujours coupler à des essais sur éprouvettes.

### 9. Caveats and Recommendations
**Quoi** : limites honnêtes + pistes d'amélioration.  
**Notre réponse** :
1. Extrapolation impossible (arbres = constante hors plage connue)
2. Biais pessimiste (RMSE estimé sur 80% des données)
3. Multicolinéarité water/superplasticizer → Feature Importance à prendre avec précaution
4. Dataset de laboratoire → variabilité chantier non modélisée
5. Recommandation : ajouter le ratio water/cement comme feature engineered

---

## Pourquoi le model card porte sur GB uniquement ?

**Cause** : l'évaluation PDF §4.2 demande "for your **best model**".  
**Comment on détermine le meilleur** : sur le RMSE **outer CV** — l'unique estimateur non biaisé. Pas le RMSE inner CV (biaisé par le tuning).

```python
best_name = min(model_results, key=lambda k: model_results[k]['rmse_scores'].mean())
# → 'Gradient Boosting'
```

---

## Questions piège à l'oral

**Q: Pourquoi pas Ridge si c'est plus interprétable ?**  
R: Ridge sacrifie 6 MPa de RMSE (10.4 vs 4.2) pour l'interprétabilité. On a quand même l'interprétabilité via les Feature Importances de GB. Si un audit légal exigeait l'interprétabilité absolue, on choisirait Ridge et on le justifierait.

**Q: Le modèle peut-il être utilisé en production ?**  
R: Non sans validation complémentaire — dataset de laboratoire, extrapolation impossible, pas de certification réglementaire.

**Q: Vos RMSE sont-ils vraiment non biaisés ?**  
R: Oui pour la nested CV outer — le test externe n'a jamais été vu pendant le tuning. Légèrement pessimistes (entraîné sur 80% des données), mais jamais optimistes.

---

## À retenir pour l'oral

> *"Une Model Card documente le modèle de manière transparente — ce qu'il fait, comment il a été évalué, où il échoue. On la fait pour le meilleur modèle (GB, RMSE = 4.2 MPa). Les 9 sections de Mitchell et al. nous forcent à être honnêtes : on doit documenter les limites (extrapolation impossible, dataset de laboratoire) autant que les performances."*
