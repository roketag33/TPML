import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pymongo import MongoClient
import redis
from sklearn.ensemble import RandomForestClassifier
import os

# Configuration de la page
st.set_page_config(page_title="Iris Classification - TPML", layout="wide", page_icon="🌺")

# --- Fonctions de chargement (Caches) ---

@st.cache_resource
def init_connections():
    """Initialise les connexions aux bases de données."""
    try:
        mongo_client = MongoClient("mongodb://localhost:27017/")
        mongo_client.server_info() # Test connexion
        db = mongo_client["tpml_iris"]
        
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping() # Test connexion
        
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
    """Entraîne un modèle Random Forest pour la démo interactive."""
    if df.empty:
        return None
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

@st.cache_resource
def load_image_model():
    """Charge le pipeline de classification d'images (ViT)."""
    try:
        from transformers import pipeline
        return pipeline("image-classification", model="google/vit-base-patch16-224")
    except Exception as e:
        return None

# --- Chargement initial des ressources ---
db, redis_client = init_connections()
df = load_data()
model = train_demo_model(df) if not df.empty else None

# --- Sidebar & Navigation ---
st.sidebar.title("🌺 Navigation")
st.sidebar.markdown("Explorez les différentes facettes du projet.")

page = st.sidebar.radio(
    "Aller vers :",
    [
        "1. 📊 Analyse Exploratoire (EDA)",
        "2. 🔮 Prédiction & Cache Redis", 
        "3. 📈 Performance & Big Data",
        "4. 📷 Vision par Ordinateur"
    ]
)

st.sidebar.markdown("---")
st.sidebar.header("Options Générales")
if st.sidebar.button("🔄 Recharger les données"):
    load_data.clear()
    st.rerun()

st.sidebar.info(
    "**Projet TPML**\n\n"
    "Architecture Polyglotte :\n"
    "- **MongoDB** : Stockage Données\n"
    "- **Cassandra** : Historique\n"
    "- **Redis** : Cache Temps Réel\n"
    "- **Spark** : Entraînement Distribué"
)

# --- Contenu Principal ---

st.title("🌺 Classification des Iris - Dashboard interactif")

if page == "1. 📊 Analyse Exploratoire (EDA)":
    st.header("📊 Analyse Exploratoire des Données (EDA)")
    
    with st.expander("📘 **Comprendre cette section (Aide)**", expanded=True):
        st.markdown("""
        **À quoi ça sert ?**
        Cette page permet de visualiser les données brutes stockées dans **MongoDB**. C'est la première étape de tout projet de Data Science : comprendre la donnée.
        
        **Ce que vous pouvez faire ici :**
        1. **Vérifier l'équilibre des classes** : Le graphique de gauche doit montrer un nombre égal de fleurs pour chaque espèce.
        2. **Analyser les corrélations** : Le scatter plot interactif à droite permet de voir quelles mesures (pétales/sépales) séparent le mieux les espèces.
        3. **Consulter les statistiques** : Le tableau en bas donne les moyennes et écarts-types.
        """)
        
    if df.empty:
        st.warning("Aucune donnée trouvée.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution des Espèces")
            st.caption("On vérifie ici que le dataset est bien équilibré.")
            fig_count = plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x='species', palette="viridis")
            st.pyplot(fig_count)
            
        with col2:
            st.subheader("Nuage de Points Interactif")
            st.caption("Jouez avec les axes pour voir comment les espèces se séparent géométriquement.")
            x_axis = st.selectbox("Choisir l'Axe X", df.columns[:-2], index=2)
            y_axis = st.selectbox("Choisir l'Axe Y", df.columns[:-2], index=3)
            
            fig_scatter = plt.figure(figsize=(6, 4))
            sns.scatterplot(data=df, x=x_axis, y=y_axis, hue='species', palette="viridis")
            st.pyplot(fig_scatter)
            
        st.subheader("Statistiques Globales")
        st.dataframe(df.describe())

elif page == "2. 🔮 Prédiction & Cache Redis":
    st.header("🔮 Prédiction Temps Réel & Cache Redis")
    
    with st.expander("📘 **Comprendre cette section (Aide)**", expanded=True):
        st.markdown("""
        **À quoi ça sert ?**
        Cette section démontre la puissance de l'architecture **Hybride ML + Redis**.
        
        **Le Défi :** Les modèles de ML peuvent être lents à répondre si beaucoup d'utilisateurs les sollicitent.
        **La Solution :** Utiliser **Redis** comme mémoire cache ultra-rapide.
        
        **Testez-le vous-même !**
        1. Réglez les sliders pour définir une fleur.
        2. Cliquez sur **Prédire**. Le modèle calcule (c'est plus long).
        3. **Re-cliquez** sans changer les valeurs. Le résultat s'affiche instantanément (< 1ms) grâce à Redis !
        """)
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.subheader("Paramètres de la Fleur")
        sl = st.slider("Longueur Sépale (cm)", 4.0, 8.0, 5.8)
        sw = st.slider("Largeur Sépale (cm)", 2.0, 5.0, 3.0)
        pl = st.slider("Longueur Pétale (cm)", 1.0, 7.0, 4.0)
        pw = st.slider("Largeur Pétale (cm)", 0.1, 3.0, 1.2)
        
        predict_btn = st.button("🚀 Lancer la Prédiction", type="primary")
        
    with col_result:
        st.subheader("Résultat de l'IA")
        if predict_btn and model:
            # Clé de cache unique basée sur les inputs
            cache_key = f"pred:{sl}:{sw}:{pl}:{pw}"
            
            # 1. Vérification dans le Cache Redis
            cached_res = redis_client.get(cache_key)
            
            if cached_res:
                st.success(f"🌿 Espèce Identifiée : **{cached_res}**")
                st.info("⚡ **HIT CACHE REDIS** : Résultat récupéré en mémoire (< 1ms).")
                st.balloons()
            else:
                # 2. Calcul par le Modèle (si pas en cache)
                prediction = model.predict([[sl, sw, pl, pw]])[0]
                st.success(f"🌿 Espèce Identifiée : **{prediction}**")
                st.warning("🧠 **MISS CACHE** : Calcul effectué par le modèle Random Forest.")
                
                # 3. Stockage dans Redis pour la prochaine fois
                redis_client.set(cache_key, prediction)

elif page == "3. 📈 Performance & Big Data":
    st.header("📈 Performances Système & Modèles Distribués")
    
    with st.expander("📘 **Comprendre cette section (Aide)**", expanded=True):
        st.markdown("""
        **À quoi ça sert ?**
        Ici, on quitte le temps réel pour analyser les travaux de fond (Batch Processing).
        
        **Ce que l'on voit :**
        1. **Résultats Spark MLlib** : La précision des modèles entraînés sur tout le Big Data. Cela prouve la qualité scientifique de l'approche.
        2. **Benchmark BDD** : Une comparaison objective entre MongoDB, Cassandra et Redis. C'est la justification technique de nos choix d'architecture.
        """)

    st.subheader("1. Résultats de la Classification Distribuée (Spark)")
    metrics_path = "output/classification/metrics.csv"
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))
        
        best_model = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
        st.markdown(f"🏆 **Champion :** Le modèle **{best_model['Model']}** est le plus performant avec une précision de **{best_model['Accuracy']:.2%}**.")
    else:
        st.error("Les métriques n'ont pas encore été générées. Lancez `classifier.py`.")
        
    st.markdown("---")
    
    st.subheader("2. Benchmark de Performance (Ops/sec)")
    bench_img = "output/benchmark/benchmark_plot.png"
    if os.path.exists(bench_img):
        st.image(bench_img, caption="Comparaison Lecture/Écriture : Redis écrase la concurrence !", use_column_width=True)
        st.info("💡 **Analyse** : Redis est ~10x à 100x plus rapide que les autres bases NoSQL pour les opérations simples, ce qui valide son utilisation en cache.")
    else:
        st.warning("Le graphique de benchmark n'est pas disponible. Lancez `benchmark_suite.py`.")

elif page == "4. 📷 Vision par Ordinateur":
    st.header("📷 Reconnaissance d'Images (Vision par Ordinateur)")
    
    with st.expander("📘 **Comprendre cette section (Aide)**", expanded=True):
        st.markdown("""
        **À quoi ça sert ?**
        C'est une fonctionnalité bonus utilisant le **Deep Learning** moderne (Transformers).
        Contrairement aux onglets précédents qui utilisaient des mesures (chiffres), ici l'IA "regarde" une photo.
        
        **Technologie :** Vision Transformer (ViT) de Google. C'est un réseau de neurones qui découpe l'image en morceaux pour l'analyser.
        
        **Essayez !** Importez une photo de fleur (téléchargée sur Google Images) et voyez si l'IA la reconnaît.
        """)
        
    uploaded_file = st.file_uploader("📥 Déposez une image de fleur ici (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        from PIL import Image
        image = Image.open(uploaded_file)
        
        col_img, col_an = st.columns(2)
        with col_img:
            st.image(image, caption='Votre image', use_column_width=True)
        
        with col_an:
            st.write("🤖 **L'IA analyse l'image...**")
            with st.spinner('Chargement du modèle Vision Transformer...'):
                classifier = load_image_model()
                if classifier:
                    predictions = classifier(image)
                    st.success("Analyse terminée !")
                    
                    # Top prédiction
                    top_p = predictions[0]
                    confidence = top_p['score']
                    label = top_p['label']
                    
                    if confidence > 0.7:
                        st.balloons()
                        st.markdown(f"### 🌸 Résultat : **{label}**")
                        st.markdown(f"**Confiance : {confidence:.1%}**")
                    else:
                        st.markdown(f"### 🤔 Résultat incertain : **{label}**")
                        st.caption(f"Confiance faible ({confidence:.1%}). L'image est peut-être floue ou ce n'est pas une fleur connue.")
                    
                    # Tableau détaillé
                    st.markdown("#### Détails des probabilités :")
                    res_data = [{"Fleur": p['label'], "Probabilité": p['score']} for p in predictions]
                    st.dataframe(pd.DataFrame(res_data).style.format({"Probabilité": "{:.2%}"}))
                else:
                    st.error("Impossible de charger le modèle de vision. Vérifiez votre connexion internet pour télécharger les poids du modèle.")
