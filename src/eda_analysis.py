import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.data_loader import load_iris_dataset

def perform_eda():
    """Exécute l'analyse exploratoire des données."""
    print("Chargement des données...")
    df = load_iris_dataset()
    
    output_dir = "output/plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Calcul des statistiques descriptives...")
    desc_stats = df.describe()
    print(desc_stats)
    with open("output/eda_stats.txt", "w") as f:
        f.write(desc_stats.to_string())
        
    print("Génération des graphiques...")
    
    # 1. Pairplot
    sns.pairplot(df, hue="species")
    plt.savefig(f"{output_dir}/pairplot.png")
    plt.close()
    
    # 2. Correlation Matrix
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Matrice de Corrélation")
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # 3. Boxplots per feature
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, orient="h")
    plt.title("Distribution des variables")
    plt.savefig(f"{output_dir}/boxplots.png")
    plt.close()

    # 4. Violin plots by species
    for col in numeric_df.columns:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x="species", y=col, data=df)
        plt.title(f"Distribution de {col} par espèce")
        plt.savefig(f"{output_dir}/violin_{col}.png")
        plt.close()

    print("EDA terminée. Rapports générés dans output/")

if __name__ == "__main__":
    perform_eda()
