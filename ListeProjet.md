Projet 1 — Classification NoSQL sur IRIS
Titre
Classification d’espèces de fleurs avec MongoDB et optimisation des performances
Dataset officiel
●
●
UCI Machine Learning Repository — Iris Dataset
https://archive.ics.uci.edu/dataset/53/iris
Kaggle — Iris Dataset
https://www.kaggle.com/datasets/uciml/iris
Contexte et problématique métier
Une application scientifique collecte des mesures morphologiques de fleurs.
L’objectif est de prédire automatiquement l’espèce de la fleur parmi :
●
●
●
Setosa
Versicolor
Virginica
Objectifs pédagogiques
Ce projet permet de mettre en œuvre :
●
●
●
●
la modélisation NoSQL orientée performance
l’optimisation des accès via indexation
l’intégration d’un pipeline de classification
l’utilisation de Spark pour un traitement distribué
Travail demandé
1. Modélisation et stockage NoSQL optimisé
Importer le dataset Iris dans MongoDB en utilisant une structure document adaptée :
{
"id": "IR001"
,
"features": {
"sepal
_
length": 5.1,
"sepal
_
width": 3.5,
"petal
_
length": 1.4,
"petal
width": 0.2
_
},
"label": "setosa"
}
Attendus :
●
●
●
schéma flexible
regroupement des features dans un sous-document
insertion en masse
2. Pipeline de classification supervisée
Entraîner un modèle de classification sur les données Iris, par exemple :
●
●
●
Random Forest
SVM
Logistic Regression
Le modèle doit permettre de prédire l’espèce à partir des variables morphologiques.
Les prédictions devront être stockées dans MongoDB.
3. Optimisation des performances
Mettre en place des stratégies d’optimisation adaptées :
●
●
●
index simple sur label
index composé sur les variables les plus discriminantes
analyse avec MongoDB Profiler
Les étudiants devront comparer les performances :
●
●
avant optimisation
après optimisation
Mesures attendues :
●
●
●
latence moyenne
temps de réponse des requêtes
throughput
4. Intégration Big Data avec Spark
Réaliser une classification distribuée avec Spark MLlib :
●
●
●
chargement des données depuis MongoDB
entraînement du modèle dans Spark
réinjection des résultats dans MongoDB
Livrables attendus
●
●
●
●
base MongoDB modélisée et optimisée
modèle de classification entraîné
rapport de profiling et benchmarks
démonstration Spark MLlib
Projet 2 — Classification des Penguins et
Benchmark Multi-NoSQL
Titre
Classification d’espèces de manchots avec MongoDB vs Cassandra et optimisation scalable
Dataset officiel
●
●
Palmer Penguins Dataset
https://allisonhorst.github.io/palmerpenguins/
Kaggle — Penguins Dataset
https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-da
ta
Contexte et problématique métier
Un institut écologique souhaite classifier automatiquement des espèces de manchots en
Antarctique à partir de mesures biométriques.
Espèces à prédire :
●
●
●
Adelie
Chinstrap
Gentoo
Objectifs pédagogiques
Ce projet vise à :
●
●
●
●
comparer deux moteurs NoSQL (document vs colonne distribuée)
analyser la scalabilité et la performance
appliquer des stratégies avancées (partitionnement, cache)
intégrer Spark pour le traitement distribué
Travail demandé
1. Classification supervisée
Variables d’entrée :
●
●
●
●
bill
_
length
mm
_
bill
_
depth
mm
_
flipper
_
length
_
body_
mass
_g
mm
Sortie attendue :
●
species
2. Modélisation multi-NoSQL
Modèle MongoDB (document)
{
"penguin
id": "P1001"
_
,
"features": {
"bill
_
length": 46.2,
"bill
_
depth": 14.5,
"flipper
_
length": 210,
"body_
mass": 5000
},
"label": "Gentoo"
,
"island": "Biscoe"
}
Modèle Cassandra (colonne distribuée)
CREATE TABLE penguins
_
by_
island (
island TEXT,
species TEXT,
penguin
_
id UUID,
bill
_
length FLOAT,
body_
mass INT,
PRIMARY KEY ((island), species, penguin
_
id)
);
3. Benchmark comparatif MongoDB vs Cassandra
Les étudiants devront mesurer :
●
●
●
●
latence moyenne
throughput
scalabilité (augmentation du volume de données)
consommation mémoire
Exemple de synthèse attendue :
Critère MongoDB Cassandr
a
Lecture ML Très bon Excellent
Scalabilité massive Moyen Excellent
Requêtes
analytiques
Excellent Bon
4. Optimisation avancée
●
●
●
partitionnement Cassandra par île
index MongoDB sur species
cache Redis pour prédictions récentes
5. Intégration Spark MLlib
●
●
●
entraînement distribué
classification batch
stockage des résultats dans MongoDB et Cassandra
Livrables attendus
●
●
●
●
modèles MongoDB + Cassandra
modèle ML de classification
benchmarks comparatifs complets
rapport technique scalabilité
Projet 3 — Projet libre : Choix technologique et
optimisation NoSQL
Titre
Étude comparative et sélection du meilleur moteur NoSQL pour une application métier
Objectif général
Les étudiants doivent concevoir une application libre basé sur une BBD noSQL, puis :
●
●
●
●
étudier plusieurs moteurs NoSQL
mesurer leurs performances
choisir la technologie la plus adaptée
justifier leur décision par des benchmarks
Travail attendu
1. Choix du cas métier et dataset
Chaque groupe sélectionne un dataset réel (fraude, santé, churn, etc.) et définit un scénario
métier.
2. Étude comparative multi-technologies
Les étudiants doivent tester au minimum 3 moteurs :
Type NoSQL Technologie
Document MongoDB
Colonne distribuée Cassandra ou
HBase
Clé-valeur (cache) Redis
Graphe (optionnel) Neo4j
3. Analyse de performance
Mesures obligatoires :
●
●
●
●
latence moyenne
throughput (req/s)
consommation mémoire
scalabilité (volume ×10)
Outils recommandés :
●
●
●
●
MongoDB Profiler
Cassandra Stress Tool
Redis Benchmark
YCSB Benchmark Framework
4. Optimisations avancées
Selon la base retenue :
●
●
●
MongoDB : index, agrégations, sharding
Cassandra/HBase : partitionnement, réplication, tuning
Redis : caching, réduction latence
5. Intégration Big Data obligatoire
Pipeline Spark MLlib :
●
●
●
lecture depuis MongoDB ou Cassandra
entraînement distribué
stockage des résultats
Livrables attendus
●
application fonctionnelle
●
●
●
étude comparative des moteurs
benchmarks détaillés
justification du choix final