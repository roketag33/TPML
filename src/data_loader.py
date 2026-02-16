import pandas as pd
from sklearn.datasets import load_iris

def load_iris_dataset():
    """Charge le dataset Iris et le retourne sous forme de DataFrame Pandas."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

def transform_to_mongodb_document(row, id_val):
    """Transforme une ligne du DataFrame en document MongoDB."""
    return {
        "id": id_val,
        "features": {
            "sepal_length": row['sepal_length'],
            "sepal_width": row['sepal_width'],
            "petal_length": row['petal_length'],
            "petal_width": row['petal_width']
        },
        "label": row['species']
    }

def transform_to_cassandra_row(row, id_val):
    """Transforme une ligne du DataFrame en format compatible Cassandra."""
    return {
        "id": id_val,
        "sepal_length": row['sepal_length'],
        "sepal_width": row['sepal_width'],
        "petal_length": row['petal_length'],
        "petal_width": row['petal_width'],
        "species": row['species']
    }

def transform_to_redis_key_value(row, id_val):
    """Transforme une ligne en clé-valeur pour Redis."""
    key = f"iris:{id_val}"
    value = {
        "sepal_length": row['sepal_length'],
        "sepal_width": row['sepal_width'],
        "petal_length": row['petal_length'],
        "petal_width": row['petal_width'],
        "species": row['species'] 
    }
    return key, value
