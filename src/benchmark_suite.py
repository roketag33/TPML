import time
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
import redis
from cassandra.cluster import Cluster
import sys
import os

# Configuration
NUM_OPERATIONS = 1000
RESULTS_FILE = "output/benchmark/results.csv"

def get_mongo_conn():
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017,localhost:27018,localhost:27019/?replicaSet=rs0")
    client = MongoClient(mongo_uri)
    return client["tpml_benchmark"]["test_col"]

def get_cassandra_conn():
    try:
        cassandra_host = os.getenv("CASSANDRA_HOST", "localhost")
        cluster = Cluster([cassandra_host])
        session = cluster.connect()
        session.execute("CREATE KEYSPACE IF NOT EXISTS tpml_bench WITH REPLICATION = { 'class' : 'SimpleStrategy', 'replication_factor' : 1 }")
        session.set_keyspace('tpml_bench')
        session.execute("CREATE TABLE IF NOT EXISTS test_table (id text PRIMARY KEY, val text)")
        return session
    except Exception as e:
        print(f"Cassandra Error: {e}")
        return None

def get_redis_conn():
    redis_host = os.getenv("REDIS_HOST", "localhost")
    return redis.Redis(host=redis_host, port=6379, decode_responses=True)

def benchmark_mongo(col):
    if col is None: return None, None
    
    # Write
    start = time.time()
    for i in range(NUM_OPERATIONS):
        col.insert_one({"_id": str(i), "val": "test"})
    write_time = time.time() - start
    
    # Read
    start = time.time()
    for i in range(NUM_OPERATIONS):
        col.find_one({"_id": str(i)})
    read_time = time.time() - start
    
    col.drop() # Clean up
    return write_time, read_time

def benchmark_cassandra(session):
    if session is None: return None, None
    
    # Write
    start = time.time()
    prepared = session.prepare("INSERT INTO test_table (id, val) VALUES (?, ?)")
    for i in range(NUM_OPERATIONS):
        session.execute(prepared, (str(i), "test"))
    write_time = time.time() - start
    
    # Read
    start = time.time()
    prepared = session.prepare("SELECT val FROM test_table WHERE id = ?")
    for i in range(NUM_OPERATIONS):
        session.execute(prepared, (str(i),))
    read_time = time.time() - start
    
    return write_time, read_time

def benchmark_redis(r):
    if r is None: return None, None
    
    # Write
    start = time.time()
    pipe = r.pipeline()
    for i in range(NUM_OPERATIONS):
        pipe.set(f"bench:{i}", "test")
    pipe.execute() # Pipeline is faster, but maybe we want row-by-row for latency?
    # Let's do row-by-row to be comparable to others (unless we use batch everywhere)
    # Re-doing row by row for fairness of "latency per op" usually
    # But for "throughput", pipeline is better. 
    # Let's stick to simple loop for raw latency.
    
    start = time.time()
    for i in range(NUM_OPERATIONS):
        r.set(f"bench:{i}", "test")
    write_time = time.time() - start

    # Read
    start = time.time()
    for i in range(NUM_OPERATIONS):
        r.get(f"bench:{i}")
    read_time = time.time() - start
    
    # Clean
    for i in range(NUM_OPERATIONS):
        r.delete(f"bench:{i}")
        
    return write_time, read_time

def main():
    print(f"=== Starting Benchmark ({NUM_OPERATIONS} ops) ===")
    results = []
    
    # Mongo
    print("Benchmarking MongoDB...")
    m_col = get_mongo_conn()
    mw, mr = benchmark_mongo(m_col)
    results.append({"Database": "MongoDB", "Operation": "Write", "Time": mw})
    results.append({"Database": "MongoDB", "Operation": "Read", "Time": mr})
    
    # Cassandra
    print("Benchmarking Cassandra...")
    c_sess = get_cassandra_conn()
    cw, cr = benchmark_cassandra(c_sess)
    results.append({"Database": "Cassandra", "Operation": "Write", "Time": cw})
    results.append({"Database": "Cassandra", "Operation": "Read", "Time": cr})
    
    # Redis
    print("Benchmarking Redis...")
    r_conn = get_redis_conn()
    rw, rr = benchmark_redis(r_conn)
    results.append({"Database": "Redis", "Operation": "Write", "Time": rw})
    results.append({"Database": "Redis", "Operation": "Read", "Time": rr})
    
    # Visualization
    df = pd.DataFrame(results)
    df["OPS"] = NUM_OPERATIONS / df["Time"] # Operations Per Second
    print("\nResults:")
    print(df)
    
    # Ensure output dir exists
    os.makedirs("output/benchmark", exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Database", y="OPS", hue="Operation")
    plt.title("Database Performance Benchmark (Ops/Sec)")
    plt.ylabel("Operations per Second")
    plt.savefig("output/benchmark/benchmark_plot.png")
    print("Plot saved to output/benchmark/benchmark_plot.png")

if __name__ == "__main__":
    main()
