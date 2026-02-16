Aide projet – Classification des fleurs Iris
Contexte
Vous êtes analyste de données pour un laboratoire de botanique.
Votre mission est d’étudier un jeu de données sur des fleurs Iris afin de :
● Comprendre leurs caractéristiques morphologiques
● Identifier des indicateurs pertinents permettant de distinguer les espèces
● Créer un modèle statistique ou prédictif pour analyser les données
Le jeu de données Iris contient
● species : Espèce (Setosa, Versicolor, Virginica)
● sepal_length : Longueur du sépale (cm)
● sepal_width : Largeur du sépale (cm)
● petal_length : Longueur du pétale (cm)
● petal_width : Largeur du pétale (cm)
Source : Ronald Fisher (1936)
Analyse exploratoire et visualisation des
données
Partie 1 – Analyse statistique descriptive
Objectifs
● Explorer le jeu de données
● Calculer les indicateurs descriptifs
● Identifier les variables les plus discriminantes entre espèces
1. Exploration du dataset
Explorer le jeu de données :
● Nombre d’observations
● Types de variables
● Valeurs manquantes
2. Statistiques descriptives
Pour chaque variable numérique, calculer :
● Moyenne
● Médiane
● Minimum et maximum
● Variance
Variables concernées :
● sepal_length
● sepal_width
● petal_length
● petal_width
3. Comptage par espèce
Compter le nombre de fleurs par espèce :
● Setosa
● Versicolor
● Virginica
4. Variables discriminantes
Identifier quelles mesures morphologiques permettent le mieux de distinguer les espèces.
Questions à traiter
1. Quelles espèces semblent surreprésentées dans le dataset ?
2. Existe-t-il des différences marquées de taille entre les espèces ?
3. Les pétales ou les sépales semblent-ils plus discriminants ?
Partie 2 – Visualisation des données
Objectifs
● Visualiser les distributions des variables
● Observer les relations entre caractéristiques
● Identifier corrélations et séparations entre espèces
1. Distribution des variables numériques
Créer des graphiques pour chaque variable :
● Histogrammes
● Boxplots
● Densité
Variables : sepal_length, sepal_width, petal_length, petal_width
2. Scatter plots
Observer les relations entre variables :
● Longueur et largeur des pétales par espèce (petal_length vs petal_width)
● Longueur et largeur des sépales par espèce (sepal_length vs sepal_width)
3. Matrice de corrélation
Construire une matrice de corrélation pour toutes les variables numériques.
Identifier les relations les plus fortes entre variables.
Questions à traiter
1. Quelles variables semblent fortement corrélées ?
2. Existe-t-il des biais visuels à prendre en compte ?
3. Quelles observations permettent de mieux distinguer les espèces ?
Régression, classification et application
interactive
Partie 3 – Régression simple et multiple
Objectifs
● Mettre en place une régression linéaire
● Étudier les relations entre variables morphologiques
● Comparer modèle simple et multiple
1. Régression linéaire simple
Mettre en place une régression pour prédire :
● petal_length à partir de sepal_length
2. Régression multiple
Étendre la régression en utilisant 2 ou 3 variables explicatives :
● sepal_length
● sepal_width
● petal_width
3. Interprétation des résultats
Analyser :
● Coefficients
● Résidus
● R²
● p-values
4. Vérification des hypothèses
Vérifier :
● Linéarité
● Normalité des résidus
● Homoscédasticité
Questions à traiter
1. Quels paramètres influencent le plus la longueur des pétales ?
2. Le modèle multiple améliore-t-il la prédiction par rapport au modèle simple ?
3. Les hypothèses de la régression sont-elles respectées ?
Partie 4 – Classification supervisée et extraction
d’indicateurs
Objectifs
● Prédire l’espèce d’une fleur
● Identifier les variables discriminantes
● Évaluer un modèle de classification
1. Variables explicatives
Sélectionner les variables morphologiques :
● sepal_length
● sepal_width
● petal_length
● petal_width
2. Méthode de classification
Proposer un modèle supervisé et justifier le choix :
● k-NN
● Arbre de décision
● Random Forest
3. Évaluation du modèle
Évaluer la performance à l’aide de :
● Précision (accuracy)
● Matrice de confusion
● Recall / F1-score
4. Prototype interactif
Créer un prototype permettant de :
● Choisir une variable à visualiser
● Afficher sa distribution (histogramme ou boxplot)
● Visualiser la corrélation avec d’autres variables
● Observer la séparation des espèces
Questions à traiter
1. Quelles espèces sont les plus difficiles à prédire et pourquoi ?
2. Quelles variables sont les plus discriminantes pour la classification ?
3. Quels indicateurs statistiques sont les plus pertinents pour le dataset Iris ?