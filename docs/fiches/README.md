# Fiches orales — Concrete Compressive Strength

*À lire dans l'ordre avant l'examen oral.*

---

| Fiche | Sujet | Points clés |
|---|---|---|
| [00](00_projet_dataset.md) | Le projet & le dataset | 8 features, physique du béton, résultats finaux |
| [01](01_erreur_generalisation.md) | Erreur de généralisation | Train error biaisé, GE, biais-variance |
| [02](02_metriques_regression.md) | Métriques de régression | RMSE, R², MSE, neg_MSE dans GridSearchCV |
| [03](03_cross_validation.md) | Cross-Validation | 5-fold, shuffle, biais pessimiste, non-indépendance |
| [04](04_hyperparametres_tuning.md) | Hyperparamètres & Tuning | θ vs λ, GridSearch vs RandomSearch, nos grilles |
| [05](05_nested_cv.md) | Nested Cross-Validation | Overtuning, 2 boucles, implémentation sklearn |
| [06](06_pipeline_data_leakage.md) | Pipeline & Data Leakage | Comment le leakage arrive, Pipeline sklearn, préfixe `model__` |
| [07](07_ridge_regression.md) | Ridge Regression | L2, StandardScaler obligatoire, coefficients standardisés |
| [08](08_cart_rf_gb.md) | CART, Random Forest, GB | Bagging vs Boosting, variance vs biais, résultats |
| [09](09_model_card.md) | Model Card (Mitchell 2019) | 9 sections, pourquoi GB, questions piège |
| [10](10_interpretabilite.md) | Interpretabilité | Prédiction vs Explication, Feature Importance, physique béton |

---

## Questions les plus probables à l'oral

1. **"Pourquoi la nested CV et pas une simple CV ?"** → Fiche 05
2. **"Comment votre Pipeline évite-t-il le data leakage ?"** → Fiche 06
3. **"Pourquoi RMSE et pas MSE ?"** → Fiche 02
4. **"Pourquoi Gradient Boosting est meilleur que Random Forest ?"** → Fiche 08
5. **"Pourquoi avez-vous exclu les réseaux de neurones ?"** → Fiche 10
6. **"Que signifie le biais pessimiste de votre RMSE ?"** → Fiches 01 + 03
7. **"Expliquez les 9 sections de Mitchell et al."** → Fiche 09
8. **"Que vaut concrètement votre RMSE de 4.2 MPa ?"** → Fiche 00 + 02
9. **"Pourquoi shuffle=True dans votre KFold ?"** → Fiche 03
10. **"Qu'est-ce que l'overtuning et comment l'évitez-vous ?"** → Fiche 04 + 05
