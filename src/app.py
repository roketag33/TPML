import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
import redis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import json

# Configuration de la page
st.set_page_config(page_title="Iris Classification - TPML", layout="wide")

# --- Fonctions de chargement ---

@st.cache_resource
def init_connections():
    """Initialise les connexions aux bases de données."""
    try:
        mongo_client = MongoClient("mongodb://localhost:27017/")
        # Test connexion
        mongo_client.server_info()
        db = mongo_client["tpml_iris"]
        
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        
        return db, redis_client
    except Exception as e:
        st.error(f"Erreur de connexion BDD: {e}")
        return None, None

@st.cache_data
def load_data():
    """Charge les données depuis MongoDB."""
    db, _ = init_connections()
    if db is not None:
        collection = db["iris_data"]
        data = list(collection.find())
        if not data:
            return pd.DataFrame()
        
        # Aplatir la structure si nested ou récupérer les champs direct
        # Notre loader met id, features: {...}, label
        # On doit aplatir features
        normalized_data = []
        for doc in data:
            item = doc.get('features', {})
            item['species'] = doc.get('label')
            item['id'] = doc.get('id')
            normalized_data.append(item)
            
        return pd.DataFrame(normalized_data)
    return pd.DataFrame()

@st.cache_resource
def train_demo_model(df):
    """Entraîne un modèle léger pour la démo interactive."""
    if df.empty:
        return None
    
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

# --- Interface Utilisateur ---

st.title("🌺 Classification des Iris - TPML Dashboard")
st.markdown("Projet polyglotte : MongoDB, Cassandra, Redis & Spark MLlib")

# Sidebar
st.sidebar.header("Options")
refresh = st.sidebar.button("Recharger les données")
if refresh:
    load_data.clear()
    st.rerun()

# Chargement
db, redis_client = init_connections()
df = load_data()

if df.empty:
    st.warning("Aucune donnée trouvée dans MongoDB. Veuillez lancer le data_loader.")
else:
    model = train_demo_model(df)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Analyse Exploratoire (EDA)", "🔮 Prédiction & Cache", "📈 Performance Spark"])
    
    with tab1:
        st.header("Analyse Exploratoire des Données")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution par Espèce")
            fig_count = plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x='species', palette="viridis")
            st.pyplot(fig_count)
            
        with col2:
            st.subheader("Scatter Plot Interactif")
            x_axis = st.selectbox("Axe X", df.columns[:-2], index=2) # petal_length par defaut
            y_axis = st.selectbox("Axe Y", df.columns[:-2], index=3) # petal_width par defaut
            
            fig_scatter = plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='species', palette="viridis")
            st.pyplot(fig_scatter)
            
        st.subheader("Statistiques Descriptives")
        st.dataframe(df.describe())

    with tab2:
        st.header("Simulation de Prédiction (avec Cache Redis)")
        
        col_input, col_result = st.columns([1, 1])
        
        with col_input:
            sl = st.slider("Sepal Length", 4.0, 8.0, 5.8)
            sw = st.slider("Sepal Width", 2.0, 5.0, 3.0)
            pl = st.slider("Petal Length", 1.0, 7.0, 4.0)
            pw = st.slider("Petal Width", 0.1, 3.0, 1.2)
            
            predict_btn = st.button("Prédire l'espèce")
            
        with col_result:
            if predict_btn and model:
                # Clé de cache
                cache_key = f"pred:{sl}:{sw}:{pl}:{pw}"
                
                # Check Redis
                cached_res = redis_client.get(cache_key)
                
                if cached_res:
                    st.success(f"Espèce prédite : **{cached_res}**")
                    st.info("ℹ️ Résultat récupéré depuis le **Cache Redis** (Latence < 1ms)")
                else:
                    # Prediction
                    prediction = model.predict([[sl, sw, pl, pw]])[0]
                    st.success(f"Espèce prédite : **{prediction}**")
                    st.warning("⚠️ Résultat calculé par le modèle (non caché)")
                    
                    # Mise en cache
                    redis_client.set(cache_key, prediction)

    with tab3:
        st.header("Performances du Modèle Distribué (Spark MLlib)")
        
        metrics_path = "output/classification/metrics.csv"
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
            
            # Afficher le meilleur modèle
            best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
            st.write(f"🏆 Meilleur Modèle : **{best_model['Model']}** avec {best_model['Accuracy']:.4f} d'accuracy.")
        else:
            st.error("Fichier de métriques introuvable. Veuillez exécuter classifier.py.")
            
        st.markdown("---")
        st.header("Benchmark Bases de Données")
        
        bench_img = "output/benchmark/benchmark_plot.png"
        if os.path.exists(bench_img):
            st.image(bench_img, caption="Comparaison des performances (Opérations/seconde)", use_column_width=True)
        else:
            st.info("Pas d'image de benchmark trouvée.")
