import pandas as pd
from sklearn.datasets import load_iris
from pymongo import MongoClient
import redis
from cassandra.cluster import Cluster
import time

def load_iris_dataset():
    """Charge le dataset Iris depuis le fichier local 'data_source/iris.data'."""
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    try:
        df = pd.read_csv('data_source/iris.data', header=None, names=column_names)
        # Nettoyage des lignes vides potentielles
        df.dropna(inplace=True)
        return df
    except FileNotFoundError:
        print("Erreur: Le fichier 'data_source/iris.data' est introuvable.")
        return pd.DataFrame()

def transform_to_mongodb_document(row, id_val):
    """Transforme une ligne du DataFrame en document MongoDB."""
    return {
        "id": id_val,
        "features": {
            "sepal_length": float(row['sepal_length']),
            "sepal_width": float(row['sepal_width']),
            "petal_length": float(row['petal_length']),
            "petal_width": float(row['petal_width'])
        },
        "label": str(row['species'])
    }

def transform_to_cassandra_row(row, id_val):
    """Transforme une ligne du DataFrame en format compatible Cassandra."""
    return {
        "id": id_val,
        "sepal_length": float(row['sepal_length']),
        "sepal_width": float(row['sepal_width']),
        "petal_length": float(row['petal_length']),
        "petal_width": float(row['petal_width']),
        "species": str(row['species'])
    }

def transform_to_redis_key_value(row, id_val):
    """Transforme une ligne en clé-valeur pour Redis."""
    key = f"iris:{id_val}"
    value = {
        "sepal_length": float(row['sepal_length']),
        "sepal_width": float(row['sepal_width']),
        "petal_length": float(row['petal_length']),
        "petal_width": float(row['petal_width']),
        "species": str(row['species'])
    }
    return key, value

def main():
    print("=== Démarrage du Data Loader Polyglotte ===")
    
    # 1. Connexion aux bases de données
    print("Connexion à MongoDB...")
    mongo_client = MongoClient("mongodb://localhost:27017/")
    mongo_db = mongo_client["tpml_iris"]
    mongo_col = mongo_db["iris_data"]
    
    # Retry logic for Cassandra as it takes time to start
    print("Connexion à Cassandra (tentatives)...")
    cassandra_session = None
    for i in range(10):
        try:
            cluster = Cluster(['localhost'])
            cassandra_session = cluster.connect()
            print("Cassandra connecté !")
            break
        except Exception as e:
            print(f"Cassandra non prêt, attente... ({e})")
            time.sleep(5)
            
    if cassandra_session:
        cassandra_session.execute("""
            CREATE KEYSPACE IF NOT EXISTS tpml_g2 
            WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 1 }
        """)
        cassandra_session.set_keyspace('tpml_g2')
        cassandra_session.execute("""
            CREATE TABLE IF NOT EXISTS iris (
                id text PRIMARY KEY,
                sepal_length float,
                sepal_width float,
                petal_length float,
                petal_width float,
                species text
            )
        """)
    else:
        print("ECHEC connexion Cassandra - Insertion annulée pour Cassandra")

    print("Connexion à Redis...")
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

    # 2. Chargement et Transformation
    df = load_iris_dataset()
    print(f"Chargement de {len(df)} entrées...")

    # 3. Insertion dans les bases
    mongo_docs = []
    
    for index, row in df.iterrows():
        id_val = f"IR{index:03d}"
        
        # MongoDB
        mongo_docs.append(transform_to_mongodb_document(row, id_val))
        
        # Cassandra
        if cassandra_session:
            c_row = transform_to_cassandra_row(row, id_val)
            query = """
                INSERT INTO iris (id, sepal_length, sepal_width, petal_length, petal_width, species)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cassandra_session.execute(query, (
                c_row['id'], c_row['sepal_length'], c_row['sepal_width'], 
                c_row['petal_length'], c_row['petal_width'], c_row['species']
            ))
            
        # Redis
        r_key, r_val = transform_to_redis_key_value(row, id_val)
        redis_client.hset(r_key, mapping=r_val)

    # Bulk insert Mongo
    if mongo_docs:
        mongo_col.delete_many({}) # Clean before insert
        mongo_col.insert_many(mongo_docs)
        print(f"MongoDB: {len(mongo_docs)} documents insérés.")

    print(f"Redis: {len(df)} clés insérées.")
    print("=== Chargement terminé avec succès ===")

if __name__ == "__main__":
    main()
