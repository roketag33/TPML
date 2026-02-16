from pymongo import MongoClient
import time
import pprint

def run_profiling():
    print("=== Analyse de Performance MongoDB (Profiling) ===")
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["tpml_iris"]
    
    # 1. Reset & Activer
    print("1. Configuration du Profiling...")
    # D'abord désactiver pour pouvoir drop (si actif)
    db.command("profile", 0)
    db.system.profile.drop()
    
    # Activer le Profiling (Level 2)
    db.command("profile", 2)
    
    # 2. Exécuter des requêtes types
    print("2. Exécution de requêtes de test...")
    col = db["iris_data"]
    
    # Requête A : Filtrer par espèce (Indexé)
    print("   - Requête A : Find 'Iris-setosa'")
    list(col.find({"label": "Iris-setosa"}))
    
    # Requête B : Filtrer par dimensions (Index Composé)
    print("   - Requête B : Find Petal Length > 1.5 & Width < 0.5")
    list(col.find({"features.petal_length": {"$gt": 1.5}, "features.petal_width": {"$lt": 0.5}}))
    
    # Requête C : Agrégation complexe (Non indexée, probablement)
    print("   - Requête C : Agrégation Moyenne par espèce")
    pipeline = [
        {"$group": {"_id": "$label", "avg_sepal_length": {"$avg": "$features.sepal_length"}}}
    ]
    list(col.aggregate(pipeline))

    # 3. Analyser les logs
    print("\n3. Analyse des logs de profiling :")
    time.sleep(1) # Laisser le temps d'écrire
    
    profiling_data = list(db.system.profile.find().sort("ts", -1).limit(5))
    
    for entry in profiling_data:
        op = entry.get('op')
        ns = entry.get('ns')
        query = entry.get('command') or entry.get('query')
        duration = entry.get('millis')
        plan = entry.get('planSummary')
        exec_stats = entry.get('execStats', {})
        docs_examined = entry.get('docsExamined') or exec_stats.get('totalDocsExamined')
        
        print(f"\n--- Opération : {op} ({duration} ms) ---")
        print(f"Namespace: {ns}")
        print(f"Query: {query}")
        print(f"Plan: {plan}")
        print(f"Docs Examined: {docs_examined}")

    # 4. Désactiver le profiling (Eviter de saturer les logs)
    print("\n4. Désactivation du Profiling...")
    db.command("profile", 0)
    print("=== Fin de l'analyse ===")

if __name__ == "__main__":
    run_profiling()
