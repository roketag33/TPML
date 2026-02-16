from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import pandas as pd
import os

def get_spark_session():
    """Initialise la session Spark avec le connecteur MongoDB."""
    return SparkSession.builder \
        .appName("TPML_Iris_Classification") \
        .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/tpml_iris.iris_data") \
        .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/tpml_iris.iris_predictions") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.4.0") \
        .getOrCreate()

def train_and_evaluate(model, train_data, test_data, model_name):
    """Entraîne et évalue un modèle."""
    print(f"\n--- Entraînement de {model_name} ---")
    fitted_model = model.fit(train_data)
    predictions = fitted_model.transform(test_data)
    
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction", metricName="f1")
    
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}")
    print(f"{model_name} - F1 Score: {f1:.4f}")
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "F1 Score": f1
    }

def main():
    spark = get_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    
    print("Chargement des données depuis MongoDB...")
    # Lecture depuis MongoDB
    df = spark.read.format("mongodb").load()
    
    print("=== Schema ===")
    df.printSchema()
    
    print("=== First 5 rows ===")
    df.show(5)
    
    
    # Arrêt temporaire pour debug - COMMENTÉ POUR ACTIVER LA SUITE
    # spark.stop()
    # return

    # Flatten features
    df = df.select(
        "label",
        "features.sepal_length",
        "features.sepal_width",
        "features.petal_length",
        "features.petal_width"
    )
    
    # Indexation du label (String -> Index)
    label_indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
    
    # Assemblage des features
    assembler = VectorAssembler(
        inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"],
        outputCol="features_vec"
    )
    
    pipeline_prep = Pipeline(stages=[label_indexer, assembler])
    prepared_data = pipeline_prep.fit(df).transform(df)
    
    # Split Train/Test
    train_data, test_data = prepared_data.randomSplit([0.8, 0.2], seed=42)
    print(f"Train size: {train_data.count()} | Test size: {test_data.count()}")
    
    results = []
    
    # 1. Random Forest
    rf = RandomForestClassifier(labelCol="labelIndex", featuresCol="features_vec", numTrees=10, seed=42)
    results.append(train_and_evaluate(rf, train_data, test_data, "Random Forest"))
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier(labelCol="labelIndex", featuresCol="features_vec", seed=42)
    results.append(train_and_evaluate(dt, train_data, test_data, "Decision Tree"))
    
    # 3. Logistic Regression
    lr = LogisticRegression(labelCol="labelIndex", featuresCol="features_vec", maxIter=10)
    results.append(train_and_evaluate(lr, train_data, test_data, "Logistic Regression"))
    
    # Export des résultats
    results_df = pd.DataFrame(results)
    print("\n=== Résumé des Performances ===")
    print(results_df)
    
    output_dir = "output/classification"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/metrics.csv", index=False)
    
    spark.stop()

if __name__ == "__main__":
    main()
