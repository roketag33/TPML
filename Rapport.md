





Classification d'Espèces de Fleurs
avec MongoDB et Optimisation des Performances

Rapport de Projet — Machine Learning & Bases de Données NoSQL




Module : Big Data & Machine Learning
Date : 16/02/2026

Étudiants : Sarrazin Alexandre
Établissement : Sup de Vinci
Table des Matières
[Clic droit → Mettre à jour les champs pour générer la table des matières dans Word]




Introduction
Dans le contexte actuel de la science des données, la capacité à extraire des connaissances à partir de données structurées constitue une compétence fondamentale. Le présent projet s'inscrit dans cette démarche en abordant un problème classique mais riche d'enseignements : la classification automatique d'espèces de fleurs d'Iris à partir de leurs caractéristiques morphologiques.
Le jeu de données Iris, publié par Ronald Fisher en 1936 et hébergé sur le UCI Machine Learning Repository, constitue l'un des benchmarks les plus utilisés en apprentissage automatique. Il comprend 150 observations réparties équitablement entre trois espèces (Iris-setosa, Iris-versicolor et Iris-virginica), chacune décrite par quatre mesures numériques : la longueur et la largeur des sépales, ainsi que la longueur et la largeur des pétales.
Problématique
Une application scientifique collecte des mesures morphologiques de fleurs. L'objectif est de concevoir un système capable de prédire automatiquement l'espèce d'une fleur parmi les trois espèces connues, en s'appuyant sur une architecture technique performante et scalable. Ce projet soulève plusieurs questions : quelles variables sont les plus discriminantes ? Quel modèle de classification offre les meilleures performances ? Comment architecturer le stockage des données et des prédictions pour garantir des temps de réponse optimaux ?
Objectifs du projet
Ce projet poursuit plusieurs objectifs complémentaires. Premièrement, réaliser une analyse exploratoire complète du dataset afin d'identifier les patterns statistiques et les variables discriminantes. Deuxièmement, modéliser les relations entre variables via des régressions linéaires (simple et multiple) pour quantifier l'influence de chaque caractéristique. Troisièmement, entraîner et comparer plusieurs modèles de classification supervisée à l'aide d'Apache Spark MLlib. Quatrièmement, concevoir une architecture NoSQL polyglotte combinant MongoDB, Cassandra et Redis, chacune optimisée pour un cas d'usage spécifique. Cinquièmement, optimiser les performances via l'indexation et le profiling. Enfin, développer un prototype interactif sous forme de dashboard Streamlit intégrant le cache Redis pour des prédictions en temps réel.
Stack technique
L'ensemble du projet repose sur Python 3.13 comme langage principal, avec les bibliothèques pandas, scikit-learn, statsmodels, seaborn et matplotlib pour l'analyse et la visualisation. Apache Spark MLlib (PySpark 3.5.1) est utilisé pour l'entraînement distribué des modèles de classification. Le stockage s'appuie sur trois bases NoSQL conteneurisées via Docker : MongoDB (document store), Cassandra (wide-column store) et Redis (cache in-memory). Le MongoDB Spark Connector 10.3.0 assure l'intégration entre Spark et MongoDB. Enfin, Streamlit est utilisé pour le dashboard interactif.

Partie 1 : Analyse Exploratoire des Données
1.1 Exploration du dataset
L'exploration initiale du jeu de données constitue une étape indispensable pour comprendre la structure, la qualité et les caractéristiques des données avant toute modélisation. Le dataset Iris présente les caractéristiques suivantes :
Caractéristique
Valeur
Nombre d'observations
150
Nombre de variables
5 (4 numériques + 1 catégorielle)
Types de variables
float64 (×4), object (×1)
Valeurs manquantes
0 (aucune)
Classes
3 espèces (Iris-setosa, Iris-versicolor, Iris-virginica)


Le dataset ne présente aucune valeur manquante, ce qui élimine le besoin d'imputation et garantit une analyse sur des données complètes. Les quatre variables numériques représentent des mesures morphologiques en centimètres, tandis que la variable catégorielle identifie l'espèce de chaque observation.
1.2 Statistiques descriptives globales
Le tableau ci-dessous présente les statistiques descriptives des quatre variables numériques, permettant d'appréhender la distribution et la dispersion des mesures dans l'ensemble du dataset :
Statistique
sepal_length
sepal_width
petal_length
petal_width
Moyenne
5.843
3.057
3.758
1.199
Écart-type
0.828
0.436
1.765
0.762
Minimum
4.300
2.000
1.000
0.100
Q1 (25%)
5.100
2.800
1.600
0.300
Médiane (50%)
5.800
3.000
4.350
1.300
Q3 (75%)
6.400
3.300
5.100
1.800
Maximum
7.900
4.400
6.900
2.500


Plusieurs observations ressortent de ces statistiques. La variable petal_length présente l'écart-type le plus élevé (1.765), ce qui traduit une très forte dispersion liée aux différences inter-espèces. En revanche, sepal_width affiche la dispersion la plus faible (0.436), suggérant une moindre variabilité entre les espèces. L'écart entre le premier quartile (1.600) et le troisième quartile (5.100) de petal_length est particulièrement marqué, révélant une distribution bimodale caractéristique de la séparation entre Setosa et les deux autres espèces.
1.3 Répartition par espèce
L'analyse de la répartition des observations par espèce est essentielle pour évaluer l'équilibre du dataset, un facteur déterminant pour la qualité de la classification.
Espèce
Nombre d'observations
Proportion
Iris-setosa
50
33.3%
Iris-versicolor
50
33.3%
Iris-virginica
50
33.3%


Le dataset est parfaitement équilibré avec exactement 50 observations par espèce, soit une proportion identique de 33.3% pour chacune. Cette répartition homogène est un atout majeur pour la classification supervisée, car elle élimine le risque de biais en faveur d'une classe surreprésentée. Aucune technique de rééquilibrage (oversampling ou undersampling) n'est donc nécessaire.
1.4 Moyennes par espèce
La comparaison des moyennes par espèce permet d'identifier les variables qui différencient le plus nettement les trois groupes :
Espèce
sepal_length
sepal_width
petal_length
petal_width
Iris-setosa
5.006
3.418
1.464
0.244
Iris-versicolor
5.936
2.770
4.260
1.326
Iris-virginica
6.588
2.974
5.552
2.026


Les différences entre espèces sont flagrantes sur les dimensions des pétales. Iris-setosa se distingue radicalement avec des pétales très courts (1.464 cm en moyenne) et très étroits (0.244 cm), tandis que Iris-virginica présente des pétales nettement plus grands (5.552 cm de long, 2.026 cm de large). Sur les sépales, les écarts sont moins prononcés : la longueur des sépales varie de 5.006 cm (Setosa) à 6.588 cm (Virginica), un écart relatif bien moindre.
1.5 Analyse de la variance par espèce
Espèce
sepal_length
sepal_width
petal_length
petal_width
Iris-setosa
0.124
0.145
0.030
0.011
Iris-versicolor
0.266
0.098
0.221
0.039
Iris-virginica
0.404
0.104
0.305
0.075


L'analyse de la variance intra-espèce révèle des profils de variabilité très contrastés. Iris-setosa présente la variance la plus faible sur les dimensions des pétales (0.030 pour petal_length, 0.011 pour petal_width), ce qui traduit une espèce morphologiquement très homogène et facilement identifiable. À l'opposé, Iris-virginica affiche les variances les plus élevées (0.404 pour sepal_length, 0.305 pour petal_length), indiquant une plus grande diversité morphologique au sein de cette espèce. Cette variabilité plus importante chez Virginica explique en partie les confusions observées avec Versicolor lors de la classification.
1.6 Variables discriminantes
L'identification des variables les plus discriminantes est cruciale pour comprendre quelles caractéristiques morphologiques permettent de distinguer les espèces. L'analyse comparative des écarts inter-espèces fournit des résultats sans ambiguïté.
Les pétales sont nettement plus discriminants que les sépales. L'écart de petal_length entre Setosa (1.464 cm) et Virginica (5.552 cm) atteint 4.088 cm, soit un ratio de ×3.8. L'écart de petal_width est encore plus frappant : de 0.244 cm (Setosa) à 2.026 cm (Virginica), soit un ratio de ×8.3. En comparaison, l'écart de sepal_length n'est que de 1.582 cm (ratio ×1.3), et sepal_width s'avère être la variable la moins discriminante avec une variance inter-espèces faible et des moyennes relativement proches.
Cette hiérarchie des variables — petal_width > petal_length > sepal_length > sepal_width — en termes de pouvoir discriminant est cohérente avec la littérature sur le dataset Iris et orientera le choix des features dans les modèles de classification.

Partie 2 : Visualisation des Données
La visualisation constitue un complément indispensable aux statistiques descriptives. Elle permet d'appréhender les distributions, les corrélations et les clusters de manière intuitive, et de déceler des patterns qui n'apparaissent pas dans les chiffres bruts.
2.1 Distribution des variables
Boxplots
Les boxplots offrent une vision synthétique de la distribution de chaque variable, en représentant la médiane, les quartiles et les valeurs extrêmes. Ils permettent de comparer visuellement les plages de variation et d'identifier d'éventuels outliers.
![Boxplots](file:///Users/alexandre/Documents/dev/TPML/output/plots/boxplots.png)
Figure 1 — Boxplots des quatre variables morphologiques

Les boxplots confirment la forte dispersion de petal_length et petal_width par rapport aux variables sépales. On observe également que sepal_width est la seule variable présentant un léger outlier vers les valeurs basses, autour de 2.0 cm.
Violin plots par espèce
Les violin plots enrichissent l'analyse en combinant un boxplot avec une estimation de la densité de probabilité, permettant de visualiser la forme de la distribution pour chaque espèce.
![Violin plot de sepal_length](file:///Users/alexandre/Documents/dev/TPML/output/plots/violin_sepal_length.png)
Figure 2 — Violin plot de sepal_length par espèce

![Violin plot de sepal_width](file:///Users/alexandre/Documents/dev/TPML/output/plots/violin_sepal_width.png)
Figure 3 — Violin plot de sepal_width par espèce

![Violin plot de petal_length](file:///Users/alexandre/Documents/dev/TPML/output/plots/violin_petal_length.png)
Figure 4 — Violin plot de petal_length par espèce

![Violin plot de petal_width](file:///Users/alexandre/Documents/dev/TPML/output/plots/violin_petal_width.png)
Figure 5 — Violin plot de petal_width par espèce

Les violin plots de petal_length et petal_width montrent des distributions clairement séparées entre les trois espèces, avec Setosa formant un cluster très compact dans les valeurs basses. Les distributions de Versicolor et Virginica se chevauchent partiellement sur petal_length (zone entre 4.5 et 5.0 cm environ), ce qui explique les erreurs de classification observées entre ces deux espèces. Les violin plots de sepal_width révèlent un phénomène intéressant : Setosa a les sépales les plus larges en moyenne (3.418 cm), contrairement à l'intuition qui pourrait associer les plus grandes dimensions à Virginica.
2.2 Scatter plots et Pairplot
Le pairplot constitue l'outil de visualisation le plus complet pour explorer les relations bivariées entre toutes les combinaisons de variables, avec une coloration par espèce.
![Pairplot complet](file:///Users/alexandre/Documents/dev/TPML/output/plots/pairplot.png)
Figure 6 — Pairplot complet des quatre variables morphologiques, coloré par espèce

Le pairplot révèle plusieurs informations fondamentales. Iris-setosa forme un cluster clairement isolé dans toutes les projections impliquant les pétales, confirmant son caractère morphologiquement distinct. La projection petal_length vs petal_width offre la meilleure séparation entre les trois espèces, avec des clusters bien définis. Iris-versicolor et Iris-virginica se chevauchent partiellement sur les dimensions des sépales, ce qui rend leur discrimination plus complexe dans cet espace. Les diagrammes en diagonale (histogrammes) montrent des distributions quasi-gaussiennes pour chaque espèce prise individuellement, mais bimodales ou trimodales à l'échelle globale du dataset.
2.3 Matrice de corrélation
L'étude des corrélations entre variables permet d'identifier les redondances et les relations linéaires, informations essentielles pour la sélection de features et l'interprétation des modèles de régression.


sepal_length
sepal_width
petal_length
petal_width
sepal_length
1.000
-0.109
0.872
0.818
sepal_width
-0.109
1.000
-0.421
-0.357
petal_length
0.872
-0.421
1.000
0.963
petal_width
0.818
-0.357
0.963
1.000


![Matrice de corrélation](file:///Users/alexandre/Documents/dev/TPML/output/plots/correlation_matrix.png)
Figure 7 — Heatmap de la matrice de corrélation entre variables numériques

La matrice de corrélation met en évidence une corrélation quasi-parfaite entre petal_length et petal_width (r = 0.963), ce qui signifie que ces deux variables évoluent de manière presque proportionnelle. La longueur des sépales est également fortement corrélée aux dimensions des pétales (r = 0.872 avec petal_length, r = 0.818 avec petal_width). En revanche, sepal_width présente une corrélation quasi-nulle avec sepal_length (r = -0.109) et des corrélations négatives modérées avec les pétales (-0.421 et -0.357).
Un point d'attention important concerne l'effet de Simpson : la corrélation globale entre sepal_width et les dimensions des pétales est négative, mais cette relation peut s'inverser au sein de chaque espèce. Ce phénomène illustre l'importance d'analyser les corrélations non seulement globalement mais aussi par sous-groupe, sous peine de tirer des conclusions erronées sur les relations causales entre variables.
La forte corrélation entre petal_length et petal_width implique une certaine redondance informative. Toutefois, dans un contexte de classification, cette colinéarité n'est pas problématique car les deux variables contribuent conjointement à la séparation des classes.

Partie 3 : Régression Linéaire
La modélisation par régression linéaire permet de quantifier les relations entre variables et d'évaluer l'influence de chaque caractéristique sur la longueur des pétales, variable identifiée comme l'une des plus discriminantes. Deux approches sont comparées : une régression simple et une régression multiple.
3.1 Régression Linéaire Simple
Le premier modèle examine la relation entre sepal_length (variable explicative) et petal_length (variable à expliquer). L'objectif est de déterminer dans quelle mesure la longueur des sépales permet de prédire celle des pétales.
Résultats du modèle OLS
Paramètre
Valeur
Variable dépendante
petal_length
Variable explicative
sepal_length
R²
0.760
R² ajusté
0.758
F-statistic
468.6
Prob (F-statistic)
1.04e-47
Coefficient (sepal_length)
1.858
Erreur standard
0.086
t-statistic
21.646
p-value coefficient
< 0.001
Constante
-7.101
Durbin-Watson
1.204


![Droite de régression simple](file:///Users/alexandre/Documents/dev/TPML/output/regression/regression_simple_plot.png)
Figure 8 — Droite de régression simple : petal_length en fonction de sepal_length

Le modèle de régression simple atteint un coefficient de détermination R² de 0.760, ce qui signifie que la longueur des sépales à elle seule explique 76% de la variance de la longueur des pétales. Le coefficient de régression de 1.858 indique que pour chaque centimètre supplémentaire de longueur de sépale, la longueur du pétale augmente en moyenne de 1.858 cm. La p-value extrêmement faible (< 0.001) et la F-statistic de 468.6 confirment la significativité statistique très élevée de cette relation.
La valeur de Durbin-Watson de 1.204 signale une légère autocorrélation positive des résidus (la valeur idéale étant 2.0). Cette autocorrélation peut être attribuée au fait que les observations sont regroupées par espèce, créant des patterns dans les résidus. Les tests de normalité des résidus (Prob(Omnibus) = 0.881, Prob(JB) = 0.824) confirment que les résidus suivent bien une distribution normale, validant cette hypothèse fondamentale.
3.2 Régression Linéaire Multiple
Le modèle multiple intègre trois variables explicatives — sepal_length, sepal_width et petal_width — pour prédire petal_length. L'objectif est d'évaluer le gain apporté par l'ajout de variables supplémentaires.
Paramètre
Coefficient
Erreur std.
t-statistic
p-value
Constante
-0.263
0.297
-0.883
0.379
sepal_length
0.729
0.058
12.502
< 0.001
sepal_width
-0.646
0.068
-9.431
< 0.001
petal_width
1.447
0.068
21.399
< 0.001


Métrique
Valeur
R²
0.968
R² ajusté
0.967
F-statistic
1 473
Prob (F-statistic)
6.98e-109
Durbin-Watson
1.783
Prob(Omnibus)
0.284
Prob(JB)
0.303


Le modèle multiple constitue une amélioration considérable par rapport au modèle simple. Le R² atteint 0.968, ce qui signifie que le modèle explique désormais 96.8% de la variance de petal_length, soit un gain de 20.8 points par rapport à la régression simple. L'analyse des coefficients révèle que petal_width est la variable la plus influente (coefficient = 1.447, t = 21.4), suivie de sepal_length (coefficient = 0.729, t = 12.5). Un résultat particulièrement intéressant est le coefficient négatif de sepal_width (-0.646, t = -9.4) : à dimensions de pétales et longueur de sépale constantes, des sépales plus larges sont associées à des pétales plus courts. La constante n'est pas statistiquement significative (p = 0.379), indiquant que le modèle passe par l'origine.
3.3 Comparaison des modèles
Métrique
Régression Simple
Régression Multiple
Amélioration
R²
0.760
0.968
+20.8 points
R² ajusté
0.758
0.967
+20.9 points
F-statistic
468.6
1 473
×3.1
AIC
385.1
86.82
-298.3
BIC
391.2
98.86
-292.3


Tous les critères convergent vers la supériorité du modèle multiple. L'AIC, critère pénalisant la complexité du modèle, chute de 385 à 87 (un AIC plus faible étant préférable), confirmant que l'ajout de variables est largement justifié et ne constitue pas un sur-ajustement. Le BIC, encore plus pénalisant pour le nombre de paramètres, confirme cette conclusion avec une réduction de 292 points.
3.4 Vérification des hypothèses de régression
La validité des résultats de régression repose sur le respect de quatre hypothèses fondamentales. Leur vérification rigoureuse est présentée ci-dessous :
Hypothèse
Méthode de vérification
Résultat
Linéarité
Scatter plot + R² élevé (0.968)
✓ Respectée
Normalité des résidus
Prob(JB) = 0.303, Prob(Omnibus) = 0.284, QQ-plot
✓ Respectée
Homoscédasticité
Graphique résidus vs prédictions
✓ Respectée
Absence d'autocorrélation
Durbin-Watson = 1.783
✓ Respectée


![Résidus vs valeurs prédites](file:///Users/alexandre/Documents/dev/TPML/output/regression/residuals_plot.png)
Figure 9 — Résidus vs valeurs prédites (vérification de l'homoscédasticité)

![QQ-plot des résidus](file:///Users/alexandre/Documents/dev/TPML/output/regression/residuals_qqplot.png)
Figure 10 — QQ-plot des résidus (vérification de la normalité)

Les quatre hypothèses fondamentales de la régression linéaire sont respectées pour le modèle multiple. La linéarité est attestée par le R² très élevé. La normalité des résidus est confirmée par les tests d'Omnibus et de Jarque-Bera (p-values > 0.05, on ne rejette pas l'hypothèse nulle de normalité), ainsi que par le QQ-plot qui montre des résidus alignés sur la droite théorique. L'homoscédasticité est vérifiée visuellement sur le graphique des résidus, qui ne montre pas de structure en entonnoir. Enfin, le Durbin-Watson de 1.783 est suffisamment proche de 2.0 pour exclure une autocorrélation problématique, ce qui constitue une nette amélioration par rapport au modèle simple (DW = 1.204).

Partie 4 : Classification Supervisée
4.1 Méthodologie
La classification supervisée constitue le cœur de ce projet. L'objectif est de prédire automatiquement l'espèce d'une fleur à partir de ses quatre mesures morphologiques. Trois algorithmes de classification ont été entraînés et évalués à l'aide d'Apache Spark MLlib, le framework de machine learning distribué de l'écosystème Apache Spark.
Le pipeline de classification suit les étapes suivantes. Les labels textuels des espèces sont transformés en indices numériques via un StringIndexer. Les quatre variables explicatives sont assemblées en un vecteur unique via un VectorAssembler. Le dataset est divisé en un ensemble d'entraînement (80%) et un ensemble de test (20%) avec une graine aléatoire fixée à 42 pour la reproductibilité. Trois modèles sont entraînés : Random Forest, Decision Tree et Logistic Regression.
4.2 Résultats de classification
Modèle
Accuracy
Precision
Recall
F1-Score
Random Forest
91.67%
91.67%
91.67%
91.67%
Decision Tree
91.67%
91.67%
91.67%
91.67%
Logistic Regression
100.00%
100.00%
100.00%
100.00%


Les résultats de classification montrent des performances très élevées pour les trois modèles. Random Forest et Decision Tree atteignent une accuracy identique de 91.67%, tandis que la Logistic Regression réalise une classification parfaite à 100% sur le jeu de test. Ce résultat, bien qu'excellent, doit être nuancé par la taille relativement modeste du jeu de test (24 observations seulement), qui ne permet pas de garantir une généralisation parfaite à de nouvelles données.
4.3 Matrices de confusion
Random Forest


Prédit Setosa
Prédit Versicolor
Prédit Virginica
Réel Setosa
12
0
0
Réel Versicolor
0
4
1
Réel Virginica
0
1
6


Decision Tree


Prédit Setosa
Prédit Versicolor
Prédit Virginica
Réel Setosa
12
0
0
Réel Versicolor
0
4
1
Réel Virginica
0
1
6


Logistic Regression (Meilleur modèle)


Prédit Setosa
Prédit Versicolor
Prédit Virginica
Réel Setosa
12
0
0
Réel Versicolor
0
5
0
Réel Virginica
0
0
7


4.4 Rapports de classification détaillés
Random Forest / Decision Tree
Classe
Precision
Recall
F1-Score
Support
Setosa
1.000
1.000
1.000
12
Versicolor
0.800
0.800
0.800
5
Virginica
0.857
0.857
0.857
7
Moyenne pondérée
0.917
0.917
0.917
24


Logistic Regression
Classe
Precision
Recall
F1-Score
Support
Setosa
1.000
1.000
1.000
12
Versicolor
1.000
1.000
1.000
5
Virginica
1.000
1.000
1.000
7
Moyenne pondérée
1.000
1.000
1.000
24


4.5 Analyse des erreurs de classification
L'analyse des matrices de confusion révèle un pattern récurrent dans les erreurs de classification. Iris-setosa est classée parfaitement (12/12) par tous les modèles, sans aucune erreur. Ce résultat est cohérent avec l'analyse exploratoire qui a montré que cette espèce possède des pétales significativement plus petits, la rendant facilement identifiable dans l'espace des features.
Les seules erreurs surviennent entre Iris-versicolor et Iris-virginica, ce qui est attendu puisque ces deux espèces présentent des dimensions de pétales qui se chevauchent partiellement. Le Random Forest et le Decision Tree commettent exactement les mêmes erreurs : un Versicolor classé comme Virginica et un Virginica classé comme Versicolor. Cette symétrie dans les erreurs confirme que la frontière de décision entre ces deux espèces se situe dans une zone de chevauchement morphologique, où les mesures des pétales de Versicolor (moyenne : 4.26 cm) et de Virginica (moyenne : 5.55 cm) se recouvrent.
La supériorité de la Logistic Regression s'explique par la nature des frontières de décision dans ce dataset. Les trois espèces sont séparables par des hyperplans dans l'espace à quatre dimensions, et la régression logistique modélise précisément ces frontières linéaires. Les arbres de décision, quant à eux, créent des frontières « en escalier » (perpendiculaires aux axes) qui sont moins optimales pour capturer des séparations obliques dans l'espace des features.
Concernant les indicateurs statistiques les plus pertinents pour évaluer la classification, l'accuracy fournit une mesure globale fiable dans ce cas précis car le dataset est équilibré. Le F1-score, qui combine precision et recall de manière harmonique, est l'indicateur le plus robuste en général, particulièrement pour les classes déséquilibrées. La matrice de confusion reste indispensable pour identifier les confusions spécifiques entre classes et comprendre les faiblesses du modèle.

Partie 5 : Architecture NoSQL Polyglotte
L'un des aspects les plus ambitieux de ce projet est la mise en œuvre d'une architecture de stockage polyglotte, combinant trois systèmes NoSQL complémentaires. Chaque base de données est choisie pour ses forces spécifiques, illustrant le principe selon lequel il n'existe pas de solution universelle en matière de stockage de données.
5.1 MongoDB — Stockage principal (Document Store)
MongoDB est utilisé comme base de données principale pour le stockage des données brutes et des prédictions. Une architecture **Replica Set** à 3 nœuds (`rs0`) a été déployée pour garantir la haute disponibilité et la redondance des données.
Structure de la collection iris_data
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
Le choix de regrouper les mesures morphologiques dans un sous-document « features » exploite la capacité de MongoDB à stocker des structures imbriquées. Cette organisation offre une séparation logique entre les métadonnées (id, label) et les données de mesure, facilitant les requêtes et l'indexation. L'identifiant séquentiel (IR000, IR001, ...) assure une lisibilité métier, et l'insertion en masse via insert_many() optimise le chargement des 150 documents.
Structure de la collection iris_predictions
{
  "iris_id": "IR000",
  "original_label": "Iris-setosa",
  "predicted_index": 0.0,
  "confidence": "[0.9998, 0.0002, 2.69e-21]"
}
Les prédictions sont stockées dans une collection distincte pour ne pas polluer les données brutes. Le champ confidence contient les probabilités attribuées à chaque classe, permettant d'évaluer la certitude du modèle pour chaque prédiction.
5.2 Cassandra — Stockage historique (Wide-Column Store)
Cassandra est utilisé pour le stockage historique avec un partitionnement optimisé par espèce. Son architecture wide-column est idéale pour les requêtes analytiques portant sur une espèce spécifique.
CREATE TABLE iris (
    id text,
    sepal_length float,
    sepal_width float,
    petal_length float,
    petal_width float,
    species text,
    PRIMARY KEY ((species), id)
);
Le choix de la clé primaire PRIMARY KEY ((species), id) est stratégique : species constitue la clé de partition, ce qui signifie que toutes les fleurs de la même espèce sont physiquement stockées sur le même nœud du cluster. Ce partitionnement optimise considérablement les requêtes de type WHERE species = 'Iris-setosa' en évitant les scans multi-nœuds. L'identifiant id sert de clé de clustering pour trier les données au sein de chaque partition.
5.3 Redis — Cache temps réel (In-Memory Store)
Redis joue le rôle de cache haute performance pour les prédictions du dashboard Streamlit, offrant des temps de réponse inférieurs à la milliseconde.
La stratégie de cache repose sur une clé composite construite à partir des quatre mesures de la fleur : pred:{sepal_length}:{sepal_width}:{petal_length}:{petal_width}. La valeur associée est l'espèce prédite. Cette approche garantit qu'une prédiction déjà calculée pour un jeu de mesures identique sera servie instantanément depuis le cache, sans recalcul du modèle ML. Les données sont persistantes (pas de TTL) car les prédictions sont déterministes pour un modèle donné.
5.4 Justification de l'architecture polyglotte
L'architecture polyglotte se justifie par la complémentarité des trois systèmes. MongoDB, configuré en **Replica Set (3 nœuds)**, assure la haute disponibilité et la tolérance aux pannes pour le stockage critique. Cassandra apporte la scalabilité horizontale et le partitionnement automatique. Redis fournit les performances in-memory. Cette séparation des responsabilités suit le principe du « bon outil pour le bon usage ».

Partie 6 : Optimisation des Performances
6.1 Stratégie d'indexation MongoDB
L'optimisation des requêtes MongoDB passe par la création d'index adaptés aux patterns d'accès les plus fréquents. Trois index ont été créés en complément de l'index par défaut sur _id :
Nom de l'index
Type
Colonnes ciblées
Justification
idx_label
Simple
label
Accélère les filtres par espèce
idx_petal_dims
Composé
features.petal_length + features.petal_width
Optimise les requêtes sur les features discriminantes
idx_sepal_length
Simple
features.sepal_length
Accélère les range queries sur sepal_length


6.2 Benchmark avant/après indexation
Un benchmark rigoureux a été réalisé avec 500 itérations par requête afin de mesurer l'impact réel de l'indexation sur les temps de réponse :
Requête
Sans index (ms)
Avec index (ms)
Gain
Throughput sans
Throughput avec
Filtre par espèce
0.330
0.271
+17.9%
3 034 req/s
3 695 req/s
Filtre petal dims
0.241
0.330
-36.8%
4 144 req/s
3 028 req/s
Filtre sepal_length
0.298
0.293
+1.5%
3 360 req/s
3 413 req/s


![Comparaison des performances avant et après indexation](file:///Users/alexandre/Documents/dev/TPML/output/benchmark/index_benchmark_comparison.png)
Figure 11 — Comparaison des performances avant et après indexation (500 itérations)

Les résultats du benchmark révèlent un phénomène important : sur un dataset de seulement 150 documents, l'impact de l'indexation est modéré. Le gain le plus significatif est observé sur le filtre par espèce (+17.9%), où l'index simple sur label permet au moteur de requêtes de localiser directement les documents cibles sans scanner l'ensemble de la collection.
Le résultat contre-intuitif de l'index composé sur les pétales (-36.8%) s'explique par le surcoût de traversée de l'arbre B-tree, qui est plus coûteux qu'un scan linéaire lorsque la collection tient intégralement en RAM. Ce phénomène est bien documenté dans la littérature MongoDB : les index ne deviennent véritablement avantageux qu'au-delà d'un certain volume de données (typiquement 10 000+ documents). Sur des collections volumineuses, les gains attendus seraient de l'ordre de ×10 à ×100.
6.3 Profiling MongoDB
Le profiler MongoDB (activé au Level 2 pour capturer toutes les opérations) permet de vérifier que les index sont effectivement utilisés par le planificateur de requêtes :
Requête
Plan d'exécution
Index utilisé
find({"label": "Iris-setosa"})
IXSCAN
idx_label
find({"features.petal_length": {$gt: 1.5}, ...})
IXSCAN
idx_petal_dims
Aggregation $group par label
COLLSCAN
Aucun (normal)


Le profiling confirme que les index sont correctement utilisés pour les requêtes de filtre (IXSCAN = Index Scan). Le planificateur de requêtes choisit automatiquement l'index le plus pertinent pour chaque requête. L'agrégation par $group effectue un COLLSCAN (Collection Scan), ce qui est le comportement attendu pour une opération nécessitant la lecture de tous les documents afin de les regrouper.
6.4 Benchmark multi-bases
Une comparaison des performances des trois bases NoSQL a été réalisée sur 1 000 opérations pour évaluer leurs forces respectives :
![Benchmark comparatif MongoDB vs Cassandra vs Redis](file:///Users/alexandre/Documents/dev/TPML/output/benchmark/benchmark_plot.png)
Figure 12 — Benchmark comparatif MongoDB vs Cassandra vs Redis (opérations/seconde)

Redis domine largement en termes de débit, avec des performances 10 à 100 fois supérieures à MongoDB et Cassandra grâce à son architecture entièrement in-memory. MongoDB offre un bon compromis entre flexibilité de schéma et performances, adapté au stockage principal. Cassandra, plus lent sur de petits volumes, est optimisé pour les architectures distribuées à grande échelle où ses capacités de partitionnement automatique et de réplication prennent tout leur sens.
Ces résultats justifient pleinement l'architecture polyglotte adoptée : chaque base est utilisée là où elle excelle, et la combinaison des trois offre un système à la fois performant, flexible et scalable.

Partie 7 : Intégration Apache Spark MLlib
7.1 Pipeline de traitement
L'intégration d'Apache Spark MLlib dans le projet permet de bénéficier d'un framework de machine learning distribué, capable de monter en charge sur des volumes de données bien supérieurs aux 150 observations du dataset Iris. Le pipeline complet se décompose en sept étapes.
La première étape consiste à charger les données depuis MongoDB vers un DataFrame Spark via le MongoDB Spark Connector 10.3.0. Les données sont ensuite sauvegardées en format Parquet dans un répertoire temporaire (étape de buffering, justifiée dans la section suivante). La troisième étape applique un StringIndexer pour transformer les labels textuels des espèces en indices numériques, puis un VectorAssembler pour fusionner les quatre features en un vecteur unique. Le dataset est ensuite divisé en 80% entraînement et 20% test avec une graine aléatoire fixée à 42. Les trois modèles sont entraînés et évalués. Le meilleur modèle identifié est ré-entraîné sur l'intégralité du dataset. Enfin, les 150 prédictions et leurs probabilités associées sont réinjectées dans MongoDB dans la collection iris_predictions via PyMongo.
7.2 Contournement technique : Buffer Parquet
Un bug de compatibilité entre Spark 3.5.1 et le MongoDB Spark Connector a été identifié lors du développement. L'erreur NoSuchMethodError se produisait lors de l'appel à fit() sur un DataFrame chargé directement depuis MongoDB. Ce problème, lié à une incompatibilité entre les versions des API internes, a nécessité la mise en place d'une stratégie de contournement.
La solution adoptée consiste à utiliser un buffer Parquet intermédiaire : les données sont d'abord lues depuis MongoDB, écrites en format Parquet sur le système de fichiers local, puis relues depuis Parquet pour l'entraînement. Ce format columnar binaire présente l'avantage d'être nativement supporté par Spark sans nécessiter de connecteur externe, éliminant ainsi le bug. De plus, la lecture Parquet est généralement plus performante que la lecture directe depuis MongoDB pour les opérations de ML.
7.3 Réinjection des résultats dans MongoDB
Les résultats de classification sont réinjectés dans MongoDB via PyMongo plutôt que via le Spark Connector. Ce choix pragmatique s'explique par la plus grande fiabilité de PyMongo pour les opérations d'écriture, et par le fait que le volume de données à écrire (150 documents) ne justifie pas la complexité d'une écriture distribuée via Spark. Chaque document de prédiction contient l'identifiant de la fleur, le label original, l'index prédit et le vecteur de confiance (probabilités par classe).

Partie 8 : Prototype Interactif — Dashboard Streamlit
Le dashboard Streamlit constitue la couche de présentation du projet, offrant une interface interactive pour explorer les données, réaliser des prédictions en temps réel et visualiser les performances du système.
8.1 Page Analyse Exploratoire (EDA)
La première page du dashboard est dédiée à l'analyse exploratoire des données. Elle propose un countplot montrant la distribution des espèces, un scatter plot interactif dont les axes sont configurables par l'utilisateur via des menus déroulants, et un tableau récapitulatif des statistiques globales. Cette page permet à l'utilisateur de reproduire visuellement les analyses présentées dans ce rapport.
8.2 Page Prédiction & Cache Redis
La seconde page est la plus interactive. Elle propose quatre sliders permettant à l'utilisateur de saisir les mesures morphologiques d'une fleur (sepal_length, sepal_width, petal_length, petal_width). Un clic sur le bouton « Prédire » déclenche l'exécution du modèle Random Forest, qui retourne l'espèce prédite.
L'intégration de Redis est démontrée de manière transparente. Lors de la première prédiction pour un jeu de mesures donné, le résultat est calculé par le modèle ML puis stocké dans Redis (MISS CACHE). Si l'utilisateur relance la même prédiction sans modifier les valeurs, Redis retourne le résultat en moins d'une milliseconde (HIT CACHE), ce qui est visuellement signalé par un message vert « HIT CACHE REDIS ». Cette démonstration illustre concrètement le gain de performance apporté par le cache dans un scénario de prédiction temps réel.
8.3 Page Performance & Big Data
La troisième page présente les résultats des modèles Spark MLlib sous forme de tableau avec mise en évidence visuelle (highlight) du meilleur modèle, ainsi que le graphique de benchmark multi-bases montrant les opérations par seconde de MongoDB, Cassandra et Redis. Cette page offre une vue synthétique des performances globales du système.

Conclusion
Bilan du projet
Ce projet a permis d'aborder l'intégralité de la chaîne de traitement d'un problème de classification en data science, depuis l'analyse exploratoire des données jusqu'à la mise en production d'un prototype interactif. Les principaux résultats techniques sont les suivants.
L'analyse exploratoire a confirmé que les dimensions des pétales sont les variables les plus discriminantes pour la classification des espèces d'Iris, avec des ratios inter-espèces atteignant ×8.3 pour petal_width. La régression linéaire multiple, avec un R² de 0.968, a démontré que la combinaison de trois variables explicatives permet d'expliquer la quasi-totalité de la variance de la longueur des pétales. La classification supervisée via Spark MLlib a atteint 100% d'accuracy avec la Logistic Regression, confirmant la séparabilité linéaire des espèces dans l'espace des features. L'architecture NoSQL polyglotte a démontré la complémentarité de MongoDB, Cassandra et Redis pour différents cas d'usage. Enfin, le cache Redis a réduit les temps de réponse de ~50ms à moins d'1ms pour les prédictions répétées.
Apports pédagogiques
Au-delà des résultats techniques, ce projet a permis de développer des compétences transversales essentielles : la capacité à concevoir une architecture de données adaptée aux contraintes de performance, la maîtrise de l'écosystème Apache Spark pour le machine learning distribué, et la compréhension des compromis inhérents aux systèmes NoSQL polyglotes. Le contournement technique via le buffer Parquet a également illustré l'importance de la résolution pragmatique de problèmes de compatibilité dans un environnement technologique complexe.
Perspectives d'amélioration
Plusieurs axes d'amélioration pourraient être envisagés pour aller plus loin. L'augmentation du volume de données (passage à un dataset plus volumineux) permettrait de mieux évaluer les gains d'indexation et les performances de Cassandra en mode distribué. L'ajout de modèles plus avancés (SVM, réseaux de neurones, gradient boosting) enrichirait la comparaison. La mise en place d'une validation croisée k-fold renforcerait la fiabilité de l'évaluation des modèles. Enfin, le déploiement du dashboard sur une infrastructure cloud avec un cluster Spark multi-nœuds démontrerait la scalabilité réelle de l'architecture.
