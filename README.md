# Projet Classification Iris - Big Data & NoSQL

Ce projet met en œuvre une architecture polyglotte (MongoDB, Cassandra, Redis) et un pipeline de Machine Learning distribué avec Spark MLlib pour classifier les fleurs d'Iris.

## 🚀 Installation & Démarrage

### 1. Prérequis
- Docker Desktop (lancé)
- Python 3.10+ (recommandé 3.11 ou 3.13)
- Java 17 (pour Spark)

### 2. Démarrer l'infrastructure
Lancez les conteneurs (Mongo, Cassandra, Redis, Spark) :
```bash
docker-compose up -d
```

### 3. Installer les dépendances
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🧪 Tests & Exécution

### Étape 1 : Préparation & Chargement des Données
Le projet utilise le fichier local `iris.zip`.
1. Décompressez les données :
```bash
unzip -o iris.zip -d data_source
```
2. Peuplez les bases de données :
```bash
.venv/bin/python src/data_loader.py
```

### Étape 2 : Analyse Exploratoire (EDA) & Régression
Génère les graphiques et statistiques dans le dossier `output/` :
```bash
.venv/bin/python src/eda_analysis.py
.venv/bin/python src/regression_analysis.py
```

### Étape 3 : Classification (Spark MLlib)
Entraîne les modèles (Random Forest, etc.) via Spark :
```bash
export PYSPARK_SUBMIT_ARGS="--packages org.mongodb.spark:mongo-spark-connector_2.12:10.4.0 pyspark-shell"
.venv/bin/python src/classifier.py
```

### Étape 4 : Benchmark de Performance
Compare les latences d'écriture/lecture (Mongo vs Cassandra vs Redis) :
```bash
.venv/bin/python src/benchmark_suite.py
```

### Étape 5 : Dashboard Interactif 🌺
Lance l'interface Streamlit pour visualiser les données et tester le cache Redis :
```bash
.venv/bin/streamlit run src/app.py
```
👉 Ouvrez votre navigateur sur [http://localhost:8501](http://localhost:8501)

## 📊 Résultats
- **Rapport Complet** : Voir `walkthrough.md` (dans le dossier artifacts).
- **Profiling** : Voir `profiling_report.md`.
