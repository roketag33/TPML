# DOSSIER TECHNIQUE COMPLET — Classification des Iris
# Fichier de référence pour la rédaction du rapport Word

> Ce document contient TOUTES les données techniques, chiffres, résultats, structures et analyses
> nécessaires pour rédiger un rapport complet répondant aux exigences des deux briefs :
> - `Aide projet – Classification des iris.md`
> - `ListeProjet.md` (Projet 1)

---

## 📁 CONTEXTE DU PROJET

**Titre** : Classification d'espèces de fleurs avec MongoDB et optimisation des performances

**Problématique métier** : Une application scientifique collecte des mesures morphologiques de fleurs. L'objectif est de prédire automatiquement l'espèce parmi Setosa, Versicolor et Virginica.

**Dataset** : UCI Machine Learning Repository — Iris Dataset (Ronald Fisher, 1936)
- 150 observations
- 4 variables numériques + 1 variable cible (espèce)
- Aucune valeur manquante
- Dataset parfaitement équilibré : 50 observations par espèce

**Architecture technique** :
- **MongoDB** : Stockage principal des données et des prédictions (document store)
- **Cassandra** : Stockage historique avec partitionnement par espèce (wide-column store)
- **Redis** : Cache temps réel pour les prédictions du dashboard (<1ms)
- **Apache Spark MLlib** : Entraînement distribué des modèles de classification
- **Streamlit** : Dashboard interactif
- **Python** : Langage principal (pandas, scikit-learn, statsmodels, seaborn, matplotlib)

**Stack technique complète** :
- Python 3.13
- PySpark 3.5.1
- MongoDB Spark Connector 10.3.0
- pymongo, cassandra-driver, redis
- statsmodels (régressions OLS)
- scikit-learn (métriques, rapports de classification)
- Streamlit (dashboard)
- Docker (MongoDB, Cassandra, Redis)

---

## 📊 PARTIE 1 — ANALYSE STATISTIQUE DESCRIPTIVE

### 1.1 Exploration du dataset

| Caractéristique | Valeur |
|---|---|
| Nombre d'observations | 150 |
| Nombre de variables | 5 (4 numériques + 1 catégorielle) |
| Types de variables | float64 (×4), object (×1) |
| Valeurs manquantes | 0 (aucune) |
| Classes | 3 espèces (Iris-setosa, Iris-versicolor, Iris-virginica) |

### 1.2 Statistiques descriptives globales

| Statistique | sepal_length | sepal_width | petal_length | petal_width |
|---|---|---|---|---|
| **count** | 150 | 150 | 150 | 150 |
| **mean** | 5.843 | 3.057 | 3.758 | 1.199 |
| **std** | 0.828 | 0.436 | 1.765 | 0.762 |
| **min** | 4.300 | 2.000 | 1.000 | 0.100 |
| **25%** | 5.100 | 2.800 | 1.600 | 0.300 |
| **50% (médiane)** | 5.800 | 3.000 | 4.350 | 1.300 |
| **75%** | 6.400 | 3.300 | 5.100 | 1.800 |
| **max** | 7.900 | 4.400 | 6.900 | 2.500 |

### 1.3 Comptage par espèce

| Espèce | Nombre |
|---|---|
| Iris-setosa | 50 |
| Iris-versicolor | 50 |
| Iris-virginica | 50 |

**Conclusion** : Le dataset est **parfaitement équilibré** avec exactement 50 observations par espèce. Aucune espèce n'est surreprésentée.

### 1.4 Moyennes par espèce

| Espèce | sepal_length | sepal_width | petal_length | petal_width |
|---|---|---|---|---|
| Iris-setosa | 5.006 | 3.418 | 1.464 | 0.244 |
| Iris-versicolor | 5.936 | 2.770 | 4.260 | 1.326 |
| Iris-virginica | 6.588 | 2.974 | 5.552 | 2.026 |

### 1.5 Variance par espèce

| Espèce | sepal_length | sepal_width | petal_length | petal_width |
|---|---|---|---|---|
| Iris-setosa | 0.124 | 0.145 | 0.030 | 0.011 |
| Iris-versicolor | 0.266 | 0.098 | 0.221 | 0.039 |
| Iris-virginica | 0.404 | 0.104 | 0.305 | 0.075 |

**Analyse de la variance** :
- Iris-setosa a la variance la plus faible sur les pétales → espèce très homogène
- Iris-virginica a la variance la plus élevée → espèce avec plus de variabilité morphologique
- Les pétales (longueur et largeur) montrent les écarts inter-espèces les plus marqués

### 1.6 Variables discriminantes

**Les pétales sont nettement plus discriminants que les sépales**, car :
1. L'écart de petal_length entre Setosa (1.464 cm) et Virginica (5.552 cm) est de **4.088 cm** (×3.8)
2. L'écart de petal_width entre Setosa (0.244 cm) et Virginica (2.026 cm) est de **1.782 cm** (×8.3)
3. L'écart de sepal_length n'est que de 1.582 cm (×1.3) — beaucoup moins discriminant
4. sepal_width est la variable la MOINS discriminante (variance inter-espèces faible)

---

## 📊 PARTIE 2 — VISUALISATION DES DONNÉES

### 2.1 Graphiques générés

| Graphique | Fichier | Description |
|---|---|---|
| Pairplot | `output/plots/pairplot.png` | Scatter plots de toutes les combinaisons de variables, colorées par espèce |
| Matrice de corrélation | `output/plots/correlation_matrix.png` | Heatmap des corrélations entre variables numériques |
| Boxplots | `output/plots/boxplots.png` | Distribution de chaque variable (min, Q1, médiane, Q3, max) |
| Violin sepal_length | `output/plots/violin_sepal_length.png` | Distribution de sepal_length par espèce |
| Violin sepal_width | `output/plots/violin_sepal_width.png` | Distribution de sepal_width par espèce |
| Violin petal_length | `output/plots/violin_petal_length.png` | Distribution de petal_length par espèce |
| Violin petal_width | `output/plots/violin_petal_width.png` | Distribution de petal_width par espèce |

### 2.2 Matrice de corrélation (chiffres)

|  | sepal_length | sepal_width | petal_length | petal_width |
|---|---|---|---|---|
| **sepal_length** | 1.000 | -0.109 | **0.872** | **0.818** |
| **sepal_width** | -0.109 | 1.000 | -0.421 | -0.357 |
| **petal_length** | **0.872** | -0.421 | 1.000 | **0.963** |
| **petal_width** | **0.818** | -0.357 | **0.963** | 1.000 |

**Corrélations clés** :
- **petal_length ↔ petal_width** : r = 0.963 → Corrélation **très forte** (quasi parfaite)
- **sepal_length ↔ petal_length** : r = 0.872 → Corrélation **forte**
- **sepal_length ↔ petal_width** : r = 0.818 → Corrélation **forte**
- **sepal_width ↔ sepal_length** : r = -0.109 → Corrélation **quasi-nulle** (aucun lien linéaire)
- **sepal_width** est **négativement** corrélée aux dimensions des pétales (-0.42 et -0.36)

**Observations visuelles clés** (pairplot) :
- Iris-setosa forme un **cluster clairement séparé** dans toutes les projections impliquant les pétales
- Iris-versicolor et Iris-virginica se **chevauchent partiellement** sur les dimensions des sépales
- Les dimensions des pétales permettent une **séparation quasi-parfaite** entre les 3 espèces

---

## 📈 PARTIE 3 — RÉGRESSION SIMPLE ET MULTIPLE

### 3.1 Régression Linéaire Simple : petal_length ~ sepal_length

```
                            OLS Regression Results
==============================================================================
Dep. Variable:           petal_length   R-squared:                       0.760
Model:                            OLS   Adj. R-squared:                  0.758
Method:                 Least Squares   F-statistic:                     468.6
                                        Prob (F-statistic):           1.04e-47
No. Observations:                 150
==============================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
const           -7.1014      0.507    -14.016      0.000      -8.103      -6.100
sepal_length     1.8584      0.086     21.646      0.000       1.689       2.028
==============================================================================
Omnibus:                        0.253   Durbin-Watson:                   1.204
Prob(Omnibus):                  0.881   Jarque-Bera (JB):                0.386
Skew:                          -0.082   Prob(JB):                        0.824
Kurtosis:                       2.812   Cond. No.                         43.4
==============================================================================
```

**Interprétation Régression Simple** :
- **R² = 0.760** : Le modèle explique **76% de la variance** de petal_length
- **Coefficient sepal_length = 1.858** : Pour chaque cm de sepal_length en plus, petal_length augmente de 1.858 cm
- **p-value < 0.001** : La relation est **hautement significative**
- **F-statistic = 468.6** : Le modèle global est très significatif
- **Normalité des résidus** : Prob(Omnibus) = 0.881, Prob(JB) = 0.824 → Les résidus sont **normaux** (hypothèse respectée)
- **Durbin-Watson = 1.204** : Légère auto-corrélation positive (valeur idéale = 2.0)
- Le graphique est disponible dans `output/regression/regression_simple_plot.png`

### 3.2 Régression Linéaire Multiple : petal_length ~ sepal_length + sepal_width + petal_width

```
                            OLS Regression Results
==============================================================================
Dep. Variable:           petal_length   R-squared:                       0.968
Model:                            OLS   Adj. R-squared:                  0.967
Method:                 Least Squares   F-statistic:                     1473.
                                        Prob (F-statistic):          6.98e-109
No. Observations:                 150
==============================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
const           -0.2627      0.297     -0.883      0.379      -0.850       0.325
sepal_length     0.7291      0.058     12.502      0.000       0.614       0.844
sepal_width     -0.6460      0.068     -9.431      0.000      -0.781      -0.511
petal_width      1.4468      0.068     21.399      0.000       1.313       1.580
==============================================================================
Omnibus:                        2.520   Durbin-Watson:                   1.783
Prob(Omnibus):                  0.284   Jarque-Bera (JB):                2.391
Skew:                           0.073   Prob(JB):                        0.303
Kurtosis:                       3.601   Cond. No.                         79.3
==============================================================================
```

**Interprétation Régression Multiple** :
- **R² = 0.968** : Le modèle explique **96.8% de la variance** → amélioration massive (+20.8 points vs simple)
- **Coefficients significatifs** :
  - `petal_width = 1.447` (t=21.4, p<0.001) : Variable la **plus influente** sur petal_length
  - `sepal_length = 0.729` (t=12.5, p<0.001) : Relation positive significative
  - `sepal_width = -0.646` (t=-9.4, p<0.001) : Relation **négative** significative (à sépales plus larges, pétales plus courts)
  - `const = -0.263` (p=0.379) : L'intercept n'est **pas significatif** → le modèle passe par l'origine
- **Normalité des résidus** : Prob(Omnibus) = 0.284, Prob(JB) = 0.303 → Résidus normaux ✓
- **Durbin-Watson = 1.783** : Pas d'autocorrélation problématique ✓
- Les graphiques de résidus et QQ-plot sont dans `output/regression/residuals_plot.png` et `output/regression/residuals_qqplot.png`

### 3.3 Comparaison Simple vs Multiple

| Métrique | Régression Simple | Régression Multiple |
|---|---|---|
| R² | 0.760 | **0.968** |
| R² ajusté | 0.758 | **0.967** |
| F-statistic | 468.6 | **1473** |
| AIC | 385.1 | **86.82** |
| BIC | 391.2 | **98.86** |

**Conclusion** : Le modèle multiple améliore **considérablement** la prédiction. L'AIC passe de 385 à 87 (le plus bas est meilleur). La variable `petal_width` est le facteur le plus influent.

### 3.4 Vérification des hypothèses de régression

| Hypothèse | Vérification | Résultat |
|---|---|---|
| **Linéarité** | Scatter plot + R² élevé | ✅ Respectée |
| **Normalité des résidus** | Prob(JB)=0.303, Prob(Omnibus)=0.284, QQ-plot | ✅ Respectée |
| **Homoscédasticité** | Graphique résidus vs prédictions | ✅ Respectée (dispersion uniforme) |
| **Absence d'autocorrélation** | Durbin-Watson = 1.783 | ✅ Respectée |

Graphiques disponibles :
- `output/regression/residuals_plot.png` : Graphique des résidus (homoscédasticité)
- `output/regression/residuals_qqplot.png` : QQ-plot (normalité des résidus)

---

## 🔮 PARTIE 4 — CLASSIFICATION SUPERVISÉE

### 4.1 Configuration

- **Framework** : Apache Spark MLlib (traitement distribué)
- **Variables explicatives** : sepal_length, sepal_width, petal_length, petal_width
- **Variable cible** : label (espèce) → indexée en labelIndex
- **Split** : 80% train / 20% test (seed=42)
- **Modèles testés** : Random Forest, Decision Tree, Logistic Regression

### 4.2 Résultats de classification

| Modèle | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 91.67% | 91.67% | 91.67% | 91.67% |
| Decision Tree | 91.67% | 91.67% | 91.67% | 91.67% |
| **Logistic Regression** | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

**Meilleur modèle** : Logistic Regression (100% sur le jeu de test)

### 4.3 Matrices de confusion

#### Random Forest
|  | Prédit Setosa (0) | Prédit Versicolor (1) | Prédit Virginica (2) |
|---|---|---|---|
| **Réel Setosa** | **12** | 0 | 0 |
| **Réel Versicolor** | 0 | **4** | 1 |
| **Réel Virginica** | 0 | 1 | **6** |

- Setosa : classé parfaitement (12/12)
- Versicolor : 1 erreur → confondu avec Virginica
- Virginica : 1 erreur → confondu avec Versicolor
- Total : 22/24 = 91.67%

#### Decision Tree
|  | Prédit Setosa (0) | Prédit Versicolor (1) | Prédit Virginica (2) |
|---|---|---|---|
| **Réel Setosa** | **12** | 0 | 0 |
| **Réel Versicolor** | 0 | **4** | 1 |
| **Réel Virginica** | 0 | 1 | **6** |

- Même profil d'erreur que Random Forest

#### Logistic Regression (Meilleur modèle)
|  | Prédit Setosa (0) | Prédit Versicolor (1) | Prédit Virginica (2) |
|---|---|---|---|
| **Réel Setosa** | **12** | 0 | 0 |
| **Réel Versicolor** | 0 | **5** | 0 |
| **Réel Virginica** | 0 | 0 | **7** |

- **Classification parfaite** : aucune erreur (24/24)

### 4.4 Rapports de classification détaillés (par classe)

#### Random Forest / Decision Tree (même profil)
| Classe | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Class_0 (Setosa) | 1.000 | 1.000 | 1.000 | 12 |
| Class_1 (Versicolor) | 0.800 | 0.800 | 0.800 | 5 |
| Class_2 (Virginica) | 0.857 | 0.857 | 0.857 | 7 |
| **weighted avg** | **0.917** | **0.917** | **0.917** | 24 |

#### Logistic Regression
| Classe | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Class_0 (Setosa) | 1.000 | 1.000 | 1.000 | 12 |
| Class_1 (Versicolor) | 1.000 | 1.000 | 1.000 | 5 |
| Class_2 (Virginica) | 1.000 | 1.000 | 1.000 | 7 |
| **weighted avg** | **1.000** | **1.000** | **1.000** | 24 |

### 4.5 Analyse des erreurs de classification

**Quelles espèces sont les plus difficiles à prédire ?**
- **Iris-setosa** est toujours classé **parfaitement** (100%) par tous les modèles → espèce morphologiquement très distincte (pétales beaucoup plus petits)
- **Iris-versicolor** et **Iris-virginica** sont les **plus confondues** entre elles → elles ont des dimensions de pétales proches (overlap visible dans le pairplot)
- La confusion Versicolor ↔ Virginica s'explique par des dimensions de pétales qui se chevauchent partiellement (Versicolor : 4.26 cm petal_length vs Virginica : 5.55 cm)

**Pourquoi la Logistic Regression surpasse les arbres ?**
- Les frontières de décision entre espèces sont **linéairement séparables** dans cet espace à 4 dimensions
- La régression logistique modélise précisément ces frontières linéaires
- Les arbres créent des frontières "en escalier" moins optimales pour ce type de séparation

---

## 🗃️ PARTIE 5 — MODÉLISATION ET STOCKAGE NoSQL

### 5.1 Structure document MongoDB (iris_data)

```json
{
  "id": "IR001",
  "features": {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  "label": "Iris-setosa"
}
```

**Choix de conception** :
- **Sous-document `features`** : Regroupement logique des mesures morphologiques (schéma flexible MongoDB)
- **`id` séquentiel** (IR000, IR001, ...) : Identifiant métier lisible
- **`label`** : Espèce de la fleur en clair
- **Insertion en masse** : `insert_many()` pour les 150 documents

### 5.2 Structure document MongoDB (iris_predictions)

```json
{
  "iris_id": "IR000",
  "original_label": "Iris-setosa",
  "predicted_index": 0.0,
  "confidence": "[0.9998, 0.0002, 2.69e-21]"
}
```

**Choix** : Stockage séparé des prédictions pour ne pas polluer les données brutes. Le champ `confidence` contient les probabilités par classe.

### 5.3 Schéma Cassandra (wide-column)

```sql
CREATE TABLE iris (
    id text,
    sepal_length float,
    sepal_width float,
    petal_length float,
    petal_width float,
    species text,
    PRIMARY KEY ((species), id)
);
```

**Partitionnement** : `PRIMARY KEY ((species), id)` → les données sont **partitionnées par espèce**. Toutes les fleurs de la même espèce sont sur le même nœud, ce qui optimise les requêtes de type `WHERE species = 'Iris-setosa'`.

### 5.4 Redis (Cache temps réel)

- **Fonction** : Cache des prédictions du dashboard Streamlit
- **Structure clé** : `pred:{sepal_length}:{sepal_width}:{petal_length}:{petal_width}` → la valeur est l'espèce prédite
- **TTL** : Pas de TTL (persistant)
- **Performance** : Réponse < 1ms (vs ~50ms pour un calcul ML)

---

## ⚡ PARTIE 6 — OPTIMISATION DES PERFORMANCES

### 6.1 Index MongoDB créés

| Index | Type | Colonnes | Nom |
|---|---|---|---|
| _id_ | Par défaut | `_id` | (automatique) |
| idx_label | **Simple** | `label` | Index sur l'espèce |
| idx_petal_dims | **Composé** | `features.petal_length` + `features.petal_width` | Index sur les features discriminantes |
| idx_sepal_length | Simple | `features.sepal_length` | Index sur sepal_length |

### 6.2 Benchmark Avant/Après Indexation (500 itérations par requête)

| Requête | SANS Index (ms) | AVEC Index (ms) | Gain | Throughput SANS | Throughput AVEC |
|---|---|---|---|---|---|
| Filter par espèce (label) | 0.3296 | 0.2707 | **+17.9%** | 3 034 req/s | 3 695 req/s |
| Filter par petal dims | 0.2413 | 0.3302 | -36.8% | 4 144 req/s | 3 028 req/s |
| Filter par sepal_length | 0.2976 | 0.2930 | +1.5% | 3 360 req/s | 3 413 req/s |

**Analyse des résultats** :
- Sur **150 documents**, l'impact de l'indexation est **modéré** car MongoDB peut scanner la collection entière très rapidement (tout tient en RAM)
- Le gain de +17.9% sur le filtre par espèce montre quand même l'efficacité de l'index simple
- Sur un gros volume (>10 000 docs), les gains seraient **beaucoup plus importants** (×10 à ×100)
- Le cas de l'index composé (petal dims) montre un surcoût car la lecture de l'arbre B-tree est plus coûteuse que le scan linéaire sur un si petit volume

Le graphique comparatif est disponible dans : `output/benchmark/index_benchmark_comparison.png`

### 6.3 Profiling MongoDB

Activation du profiler MongoDB (Level 2 = toutes les opérations) pour valider l'utilisation des index :

| Requête | Plan d'exécution | Index utilisé |
|---|---|---|
| `find({"label": "Iris-setosa"})` | **IXSCAN** | idx_label ✅ |
| `find({"features.petal_length": {$gt: 1.5}, ...})` | **IXSCAN** | idx_petal_dims ✅ |
| Aggregation `$group` par label | **COLLSCAN** | Aucun (normal pour une agrégation) |

**Conclusion** : Les index sont **bien utilisés** par MongoDB sur les requêtes de filtre. Les agrégations font un scan complet, ce qui est normal.

### 6.4 Benchmark multi-bases (ops/sec)

Comparaison des 3 bases NoSQL sur 1000 opérations :

Le graphique est disponible dans : `output/benchmark/benchmark_plot.png`

**Résultat attendu :**
- **Redis** : ~10x à 100x plus rapide que MongoDB/Cassandra (in-memory)
- **MongoDB** : Bon compromis flexibilité/performance
- **Cassandra** : Plus lent sur de petits volumes (optimisé pour le distribué)

→ **Justification de l'architecture polyglotte** : chaque base NoSQL a son rôle optimal.

---

## 🖥️ PARTIE 7 — PROTOTYPE INTERACTIF (Dashboard Streamlit)

### 7.1 Pages du dashboard

| Page | Fonctionnalité |
|---|---|
| **📊 Analyse Exploratoire (EDA)** | Distribution des espèces (countplot), scatter plot interactif (axes configurables), statistiques globales |
| **🔮 Prédiction & Cache Redis** | Sliders pour saisir les mesures, prédiction temps réel (Random Forest), cache Redis (HIT/MISS affiché) |
| **📈 Performance & Big Data** | Résultats Spark MLlib (tableau highlight), benchmark BDD (graphique ops/sec) |

### 7.2 Intégration Redis démontrée

1. L'utilisateur règle les 4 sliders (sepal_length, sepal_width, petal_length, petal_width)
2. Clic sur "Prédire" → le modèle Random Forest calcule (MISS CACHE)
3. Re-clic sans changer les valeurs → Redis retourne le résultat en <1ms (HIT CACHE)
4. Le message est affiché en vert "⚡ HIT CACHE REDIS" avec le résultat

---

## 🔗 PARTIE 8 — INTÉGRATION SPARK MLlib

### 8.1 Pipeline Spark

1. **Chargement** : Lecture depuis MongoDB → DataFrame Spark (via MongoDB Spark Connector 10.3.0)
2. **Buffer Parquet** : Sauvegarde temporaire en Parquet pour contourner un bug de compatibilité connector
3. **Preprocessing** : StringIndexer (label → labelIndex) + VectorAssembler (4 features → features_vec)
4. **Split** : 80% train / 20% test (seed=42)
5. **Entraînement** : 3 modèles (RF, DT, LR) avec évaluation complète
6. **Meilleur modèle** : Ré-entraîné sur 100% des données
7. **Sauvegarde MongoDB** : 150 prédictions + probabilités stockées dans `iris_predictions` via PyMongo

### 8.2 Contournement technique (à mentionner)

Un bug de compatibilité entre Spark 3.5.1 et le MongoDB Spark Connector (NoSuchMethodError dans le `fit()`) a nécessité une **stratégie tampon Parquet** :
- Lecture MongoDB → Parquet temporaire → Relecture Parquet → Training
- Les résultats sont ensuite réinjectés dans MongoDB via PyMongo (plus fiable que le connector pour l'écriture)

---

## 📋 RÉPONSES AUX QUESTIONS DU BRIEF

### Questions Partie 1 — Analyse descriptive

**Q1 : Quelles espèces semblent surreprésentées ?**
→ Aucune. Le dataset est parfaitement équilibré avec exactement 50 observations par espèce.

**Q2 : Existe-t-il des différences marquées de taille entre les espèces ?**
→ Oui, très marquées sur les pétales. Setosa a des pétales 3.8× plus courts et 8.3× plus étroits que Virginica. Les sépales varient moins (×1.3 en longueur).

**Q3 : Les pétales ou les sépales semblent-ils plus discriminants ?**
→ Les **pétales** sont nettement plus discriminants. L'écart inter-espèces est beaucoup plus grand sur petal_length (1.46→5.55) et petal_width (0.24→2.03) que sur les sépales.

### Questions Partie 2 — Visualisation

**Q1 : Quelles variables semblent fortement corrélées ?**
→ petal_length et petal_width (r=0.963, corrélation quasi-parfaite). Aussi sepal_length avec petal_length (r=0.872) et petal_width (r=0.818).

**Q2 : Existe-t-il des biais visuels ?**
→ Le principal biais est l'effet de Simpson : la corrélation globale sepal_width ↔ petal_length est négative (-0.42), mais au sein de chaque espèce, la relation peut être positive. Il faut toujours analyser les corrélations par espèce.

**Q3 : Quelles observations permettent de mieux distinguer les espèces ?**
→ Les scatter plots petal_length vs petal_width montrent la meilleure séparation. Setosa forme un cluster isolé en bas à gauche. Versicolor et Virginica sont séparables mais avec un léger chevauchement.

### Questions Partie 3 — Régression

**Q1 : Quels paramètres influencent le plus la longueur des pétales ?**
→ petal_width (coefficient 1.447, t=21.4) est le facteur le plus influent, suivi de sepal_length (0.729) et sepal_width (-0.646, relation négative).

**Q2 : Le modèle multiple améliore-t-il la prédiction ?**
→ Oui, considérablement. R² passe de 0.760 (simple) à 0.968 (multiple), soit +20.8 points. L'AIC passe de 385 à 87.

**Q3 : Les hypothèses sont-elles respectées ?**
→ Oui, toutes validées : linéarité (R² élevé), normalité des résidus (Prob(JB)=0.303), homoscédasticité (dispersion uniforme des résidus), absence d'autocorrélation (DW=1.783).

### Questions Partie 4 — Classification

**Q1 : Quelles espèces sont les plus difficiles à prédire ?**
→ Versicolor et Virginica sont les plus confondues entre elles (erreurs croisées dans RF et DT). Setosa est toujours classée parfaitement grâce à ses pétales très distinctifs.

**Q2 : Quelles variables sont les plus discriminantes pour la classification ?**
→ petal_length et petal_width. Elles permettent la meilleure séparation linéaire entre les 3 espèces.

**Q3 : Quels indicateurs statistiques sont les plus pertinents ?**
→ L'Accuracy (mesure globale), le F1-score (équilibre precision/recall, important pour les classes déséquilibrées), et la matrice de confusion (identification des confusions entre espèces).

---

## 📂 LISTE COMPLÈTE DES FICHIERS DE SORTIE

### Graphiques (pour insertion dans le Word)

| Fichier | Description |
|---|---|
| `output/plots/pairplot.png` | Pairplot complet par espèce |
| `output/plots/correlation_matrix.png` | Matrice de corrélation (heatmap) |
| `output/plots/boxplots.png` | Boxplots de toutes les variables |
| `output/plots/violin_sepal_length.png` | Violinplot sepal_length par espèce |
| `output/plots/violin_sepal_width.png` | Violinplot sepal_width par espèce |
| `output/plots/violin_petal_length.png` | Violinplot petal_length par espèce |
| `output/plots/violin_petal_width.png` | Violinplot petal_width par espèce |
| `output/regression/regression_simple_plot.png` | Droite de régression simple |
| `output/regression/residuals_plot.png` | Résidus vs prédictions (homoscédasticité) |
| `output/regression/residuals_qqplot.png` | QQ-plot (normalité résidus) |
| `output/benchmark/benchmark_plot.png` | Benchmark multi-bases (ops/sec) |
| `output/benchmark/index_benchmark_comparison.png` | Benchmark avant/après indexation |

### Données CSV

| Fichier | Description |
|---|---|
| `output/eda_stats.txt` | Statistiques descriptives complètes |
| `output/classification/metrics.csv` | Accuracy, Precision, Recall, F1 par modèle |
| `output/classification/confusion_matrix_random_forest.csv` | Matrice de confusion RF |
| `output/classification/confusion_matrix_decision_tree.csv` | Matrice de confusion DT |
| `output/classification/confusion_matrix_logistic_regression.csv` | Matrice de confusion LR |
| `output/classification/classification_report_random_forest.csv` | Rapport détaillé RF |
| `output/classification/classification_report_decision_tree.csv` | Rapport détaillé DT |
| `output/classification/classification_report_logistic_regression.csv` | Rapport détaillé LR |
| `output/regression/regression_simple_summary.txt` | Résumé OLS simple |
| `output/regression/regression_multiple_summary.txt` | Résumé OLS multiple |
| `output/benchmark/index_benchmark.csv` | Benchmark indexation (avant/après) |

### Scripts source

| Script | Rôle |
|---|---|
| `src/data_loader.py` | Chargement iris.data → MongoDB + Cassandra + Redis |
| `src/eda_analysis.py` | Analyse exploratoire (stats descriptives, graphiques) |
| `src/regression_analysis.py` | Régressions simple et multiple (statsmodels OLS) |
| `src/classifier.py` | Classification Spark MLlib (RF, DT, LR) + matrices de confusion |
| `src/create_indexes.py` | Création des index MongoDB |
| `src/profiling_mongo.py` | Profiling MongoDB (validation des index) |
| `src/benchmark_indexes.py` | Benchmark avant/après indexation |
| `src/benchmark_suite.py` | Benchmark multi-bases (MongoDB vs Cassandra vs Redis) |
| `src/app.py` | Dashboard Streamlit interactif |

---

## 📌 INSTRUCTIONS POUR CLAUDE

Ce document contient TOUT ce dont tu as besoin pour rédiger un rapport Word complet.

**Structure suggérée pour le Word :**
1. Page de garde
2. Introduction (contexte, problématique)
3. Partie 1 : Analyse Exploratoire (utiliser les stats et les graphiques)
4. Partie 2 : Visualisation (insérer les graphiques du dossier output/plots/)
5. Partie 3 : Régression (copier les résultats OLS, insérer les graphiques résidus)
6. Partie 4 : Classification supervisée (tableaux de métriques + matrices de confusion)
7. Partie 5 : Architecture NoSQL (structures MongoDB, Cassandra, Redis)
8. Partie 6 : Optimisation (index, profiling, benchmark avant/après)
9. Partie 7 : Intégration Spark MLlib (pipeline, résultats)
10. Partie 8 : Prototype interactif (dashboard Streamlit)
11. Conclusion

**Les images à insérer** sont dans le dossier `output/` du projet.
**Les données chiffrées** sont toutes dans ce document.
**Les réponses aux questions** sont dans la section "Réponses aux Questions du Brief".
