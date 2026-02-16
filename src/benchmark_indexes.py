"""
Benchmark Avant/Après Indexation MongoDB.
Compare les performances des requêtes avec et sans index.
"""
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient, ASCENDING

NUM_ITERATIONS = 500  # Nombre de répétitions par requête

def run_query_benchmark(col, query, label):
    """Mesure le temps moyen d'exécution d'une requête sur N itérations."""
    times = []
    for _ in range(NUM_ITERATIONS):
        start = time.perf_counter()
        list(col.find(query))
        elapsed = (time.perf_counter() - start) * 1000  # en ms
        times.append(elapsed)
    avg = sum(times) / len(times)
    return avg

def main():
    print(f"=== Benchmark Avant/Après Indexation ({NUM_ITERATIONS} itérations/requête) ===")
    
    client = MongoClient("mongodb://localhost:27017/")
    db = client["tpml_iris"]
    col = db["iris_data"]
    
    # Requêtes de test
    queries = {
        "Filter par espèce (label)": {"label": "Iris-setosa"},
        "Filter par petal dims": {"features.petal_length": {"$gt": 1.5}, "features.petal_width": {"$lt": 0.5}},
        "Filter par sepal_length": {"features.sepal_length": {"$gt": 6.0}},
    }
    
    results = []
    
    # --- PHASE 1 : SANS INDEX (Drop all custom indexes) ---
    print("\n📉 Phase 1 : Suppression des index personnalisés...")
    # Garder uniquement l'index _id (obligatoire)
    for idx_info in list(col.list_indexes()):
        if idx_info['name'] != '_id_':
            col.drop_index(idx_info['name'])
            print(f"   Supprimé : {idx_info['name']}")
    
    print("Exécution des requêtes SANS index...")
    for label, query in queries.items():
        avg_ms = run_query_benchmark(col, query, label)
        results.append({
            "Requête": label,
            "Phase": "SANS Index",
            "Latence Moy. (ms)": round(avg_ms, 4),
            "Throughput (req/s)": round(1000 / avg_ms, 1) if avg_ms > 0 else 0
        })
        print(f"   {label}: {avg_ms:.4f} ms")
    
    # --- PHASE 2 : AVEC INDEX ---
    print("\n📈 Phase 2 : Création des index...")
    col.create_index([("label", ASCENDING)], name="idx_label")
    print("   ✅ Index simple sur 'label'")
    col.create_index([
        ("features.petal_length", ASCENDING),
        ("features.petal_width", ASCENDING)
    ], name="idx_petal_dims")
    print("   ✅ Index composé sur 'petal_length + petal_width'")
    col.create_index([("features.sepal_length", ASCENDING)], name="idx_sepal_length")
    print("   ✅ Index simple sur 'sepal_length'")
    
    print("Exécution des requêtes AVEC index...")
    for label, query in queries.items():
        avg_ms = run_query_benchmark(col, query, label)
        results.append({
            "Requête": label,
            "Phase": "AVEC Index",
            "Latence Moy. (ms)": round(avg_ms, 4),
            "Throughput (req/s)": round(1000 / avg_ms, 1) if avg_ms > 0 else 0
        })
        print(f"   {label}: {avg_ms:.4f} ms")
    
    # --- Analyse et Export ---
    df = pd.DataFrame(results)
    print("\n=== Résultats Comparatifs ===")
    print(df.to_string(index=False))
    
    # Calcul du gain
    print("\n=== Gains de Performance ===")
    for q_label in queries.keys():
        sans = df[(df["Requête"] == q_label) & (df["Phase"] == "SANS Index")]["Latence Moy. (ms)"].values[0]
        avec = df[(df["Requête"] == q_label) & (df["Phase"] == "AVEC Index")]["Latence Moy. (ms)"].values[0]
        if sans > 0:
            gain = ((sans - avec) / sans) * 100
            print(f"   {q_label}: {gain:+.1f}% {'plus rapide' if gain > 0 else 'plus lent'} (de {sans:.4f}ms à {avec:.4f}ms)")
    
    # Export CSV
    output_dir = "output/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/index_benchmark.csv", index=False)
    
    # Graphique comparatif
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Latence
    sns.barplot(data=df, x="Requête", y="Latence Moy. (ms)", hue="Phase", ax=axes[0], palette=["#e74c3c", "#2ecc71"])
    axes[0].set_title("Latence Moyenne (ms) - Plus bas = Mieux")
    axes[0].tick_params(axis='x', rotation=15)
    
    # Throughput
    sns.barplot(data=df, x="Requête", y="Throughput (req/s)", hue="Phase", ax=axes[1], palette=["#e74c3c", "#2ecc71"])
    axes[1].set_title("Throughput (req/s) - Plus haut = Mieux")
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/index_benchmark_comparison.png", dpi=150)
    print(f"\n📊 Graphique sauvegardé dans {output_dir}/index_benchmark_comparison.png")

if __name__ == "__main__":
    main()
