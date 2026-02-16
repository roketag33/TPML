from pymongo import MongoClient, ASCENDING, DESCENDING

def create_indexes():
    print("--- Optimisation MongoDB : Création d'Index ---")
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["tpml_iris"]
        collection = db["iris_data"]
        
        # 1. Index Simple sur le label (pour filtrer par espèce rapidement)
        print("1. Création Index Simple sur 'label'...")
        collection.create_index([("label", ASCENDING)], name="idx_label")
        print("✅ Index 'idx_label' créé.")
        
        # 2. Index Composé sur les features discriminantes (Petal Length & Petal Width)
        # Utile pour des requêtes type : find({ "features.petal_length": {$gt: 2}, "features.petal_width": {$lt: 1} })
        print("2. Création Index Composé sur 'features.petal_length' + 'features.petal_width'...")
        collection.create_index([
            ("features.petal_length", ASCENDING),
            ("features.petal_width", ASCENDING)
        ], name="idx_petal_dims")
        print("✅ Index 'idx_petal_dims' créé.")
        
        # Lister les index finaux
        print("\n--- Index Actuels ---")
        for idx in collection.list_indexes():
            print(idx)
            
    except Exception as e:
        print(f"❌ Erreur lors de la création d'index : {e}")

if __name__ == "__main__":
    create_indexes()
