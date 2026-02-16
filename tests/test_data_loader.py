import pytest
import pandas as pd
from src.data_loader import load_iris_dataset, transform_to_mongodb_document, transform_to_cassandra_row, transform_to_redis_key_value

def test_load_iris_dataset():
    """Test if the dataset is loaded correctly as a DataFrame."""
    df = load_iris_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert all(col in df.columns for col in expected_columns)

def test_transform_to_mongodb_document():
    """Test transformation to MongoDB document format (nested features)."""
    row = pd.Series({
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2,
        'species': 'setosa'
    })
    doc = transform_to_mongodb_document(row, "IR001")
    
    assert doc['id'] == "IR001"
    assert doc['label'] == "setosa"
    assert 'features' in doc
    assert doc['features']['sepal_length'] == 5.1
    assert doc['features']['petal_width'] == 0.2

def test_transform_to_cassandra_row():
    """Test transformation to Cassandra row format (flat structure)."""
    row = pd.Series({
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2,
        'species': 'setosa'
    })
    cassandra_row = transform_to_cassandra_row(row, "IR001")
    
    # Cassandra uses a flat structure, maybe with specific ID
    assert cassandra_row['id'] == "IR001"
    assert cassandra_row['species'] == "setosa"
    assert cassandra_row['sepal_length'] == 5.1

def test_transform_to_redis_key_value():
    """Test transformation to Redis key-value pair."""
    row = pd.Series({
        'sepal_length': 5.1,
        'sepal_width': 3.5,
        'petal_length': 1.4,
        'petal_width': 0.2,
        'species': 'setosa'
    })
    key, value = transform_to_redis_key_value(row, "IR001")
    
    assert key == "iris:IR001"
    # Value should probably be a JSON string or a hash map for Redis
    assert isinstance(value, dict)
    assert value['species'] == 'setosa'
    assert value['sepal_length'] == 5.1
