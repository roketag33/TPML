import time
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

def init_replica_set():
    print("--- Initialisation du Replica Set MongoDB (rs0) ---")
    
    # On se connecte au noeud principal (mongo1)
    # Note: Dans Docker, ils se voient via leur nom de service.
    # Ici on accède depuis l'hôte via localhost:27017
    client = MongoClient("mongodb://localhost:27017/")
    
    config = {
        "_id": "rs0",
        "members": [
            {"_id": 0, "host": "mongo1:27017"},
            {"_id": 1, "host": "mongo2:27017"},
            {"_id": 2, "host": "mongo3:27017"}
        ]
    }

    try:
        # Check if already initialized
        status = client.admin.command("replSetGetStatus")
        print("✅ Replica Set déjà initialisé.")
    except Exception:
        print("Configuration du Replica Set en cours...")
        try:
            client.admin.command("replSetInitiate", config)
            print("✅ Replica Set initialisé avec succès !")
            print("Attente de l'élection du PRIMARY (10s)...")
            time.sleep(10)
        except Exception as e:
            print(f"❌ Erreur lors de l'init : {e}")

if __name__ == "__main__":
    init_replica_set()
