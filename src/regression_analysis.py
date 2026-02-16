import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_loader import load_iris_dataset

def perform_regression():
    """Exécute les analyses de régression (Simple et Multiple)."""
    print("Mise en place de la régression...")
    df = load_iris_dataset()
    output_dir = "output/regression"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Régression Simple : petal_length ~ sepal_length
    print("1. Régression Linéaire Simple...")
    X = df['sepal_length']
    y = df['petal_length']
    X_const = sm.add_constant(X) # Ajout de la constante (intercept)
    
    model_simple = sm.OLS(y, X_const).fit()
    
    with open(f"{output_dir}/regression_simple_summary.txt", "w") as f:
        f.write(model_simple.summary().as_text())
        
    # Plot Simple Regression
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sepal_length', y='petal_length', data=df, label='Données')
    plt.plot(df['sepal_length'], model_simple.predict(X_const), color='red', label='Régression')
    plt.title("Régression Simple : Petal Length ~ Sepal Length")
    plt.legend()
    plt.savefig(f"{output_dir}/regression_simple_plot.png")
    plt.close()
    
    # 2. Régression Multiple
    print("2. Régression Linéaire Multiple...")
    # Variables explicatives : sepal_length, sepal_width, petal_width
    X_multi = df[['sepal_length', 'sepal_width', 'petal_width']]
    y_multi = df['petal_length']
    X_multi_const = sm.add_constant(X_multi)
    
    model_multi = sm.OLS(y_multi, X_multi_const).fit()
    
    with open(f"{output_dir}/regression_multiple_summary.txt", "w") as f:
        f.write(model_multi.summary().as_text())
        
    # Validation des hypothèses (Résidus vs Prédictions)
    residuals = model_multi.resid
    predictions = model_multi.fittedvalues
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predictions, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Valeurs Prédites')
    plt.ylabel('Résidus')
    plt.title('Graphique des Résidus (Homoscédasticité)')
    plt.savefig(f"{output_dir}/residuals_plot.png")
    plt.close()
    
    # QQ Plot pour la normalité des résidus
    plt.figure(figsize=(10, 6))
    sm.qqplot(residuals, line='45', fit=True)
    plt.title('QQ Plot des Résidus (Normalité)')
    plt.savefig(f"{output_dir}/residuals_qqplot.png")
    plt.close()

    print("Régression terminée. Rapports générés dans output/regression")

if __name__ == "__main__":
    perform_regression()
