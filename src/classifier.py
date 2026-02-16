from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import pandas as pd
import os
import shutil
from pymongo import MongoClient

def get_spark_session():
    """Initialise la session Spark avec le connecteur MongoDB."""
    return SparkSession.builder \
        .appName("TPML_Iris_Classification") \
        .config("spark.mongodb.read.connection.uri", "mongodb://localhost:27017/tpml_iris.iris_data") \
        .config("spark.mongodb.write.connection.uri", "mongodb://localhost:27017/tpml_iris.iris_predictions") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
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
    try:
        # Lecture depuis MongoDB
        raw_df = spark.read.format("mongodb").load()
        
        # --- Stratégie Buffer (Contournement Bug Connecteur Spark 3.5) ---
        # On sauvegarde en Parquet temporaire pour couper le lien avec le connecteur MongoDB
        # qui cause des erreurs 'NoSuchMethodError' lors du fit()
        print("Mise en buffer Parquet (cache)...")
        buffer_path = "temp_iris_data.parquet"
        raw_df.write.mode("overwrite").parquet(buffer_path)
        
        # Relecture depuis Parquet (Clean Spark Interface)
        df = spark.read.parquet(buffer_path)
        print("Données chargées depuis le buffer.")
        
    except Exception as e:
        print(f"Erreur lors du chargement initial : {e}")
        spark.stop()
        return

    print("=== Schema ===")
    df.printSchema()
    
    # Flatten features
    df = df.select(
        "id",
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
    models = {}
    
    # 1. Random Forest
    rf = RandomForestClassifier(labelCol="labelIndex", featuresCol="features_vec", numTrees=10, seed=42)
    rf_res = train_and_evaluate(rf, train_data, test_data, "Random Forest")
    results.append(rf_res)
    models["Random Forest"] = rf
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier(labelCol="labelIndex", featuresCol="features_vec", seed=42)
    dt_res = train_and_evaluate(dt, train_data, test_data, "Decision Tree")
    results.append(dt_res)
    models["Decision Tree"] = dt
    
    # 3. Logistic Regression
    lr = LogisticRegression(labelCol="labelIndex", featuresCol="features_vec", maxIter=10)
    lr_res = train_and_evaluate(lr, train_data, test_data, "Logistic Regression")
    results.append(lr_res)
    models["Logistic Regression"] = lr
    
    # Export des résultats
    results_df = pd.DataFrame(results)
    print("\n=== Résumé des Performances ===")
    print(results_df)
    
    output_dir = "output/classification"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(f"{output_dir}/metrics.csv", index=False)

    # --- Sauvegarde des prédictions (Best Model) via PyMongo (Plus fiable) ---
    try:
        
        best_model_name = results_df.loc[results_df['Accuracy'].idxmax()]['Model']
        print(f"\n🏆 Meilleur modèle : {best_model_name}")
        
        # Prédiction sur tout le dataset
        best_estimator = models[best_model_name].fit(prepared_data)
        final_predictions = best_estimator.transform(prepared_data)
        
        # Collecte des résultats (Driver memory is fine for 150 rows)
        rows_to_save = final_predictions.select("id", "label", "prediction", "probability").collect()
        
        print(f"Sauvegarde de {len(rows_to_save)} prédictions dans MongoDB (via PyMongo)...")
        
        client = MongoClient("mongodb://localhost:27017/")
        db_mongo = client["tpml_iris"]
        col_pred = db_mongo["iris_predictions"]
        col_pred.delete_many({}) # Clean old
        
        docs = []
        for row in rows_to_save:
            docs.append({
                "iris_id": row["id"],
                "original_label": row["label"],
                "predicted_index": float(row["prediction"]),
                "confidence": str(row["probability"]) # Convert Vector to string
            })
            
        col_pred.insert_many(docs)
        print("✅ Prédictions sauvegardées avec succès !")
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde MongoDB : {e}")
    
    # Cleanup buffer
    if os.path.exists(buffer_path):
        shutil.rmtree(buffer_path)
        
    spark.stop()

if __name__ == "__main__":
    main()
