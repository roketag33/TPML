# --- Étape 1 : Récupérer Java depuis une image officielle stable ---
FROM eclipse-temurin:17-jre as java-stage

# --- Étape 2 : Image Principale Python ---
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier Java depuis l'étape 1 (Plus robuste que apt-get)
COPY --from=java-stage /opt/java/openjdk /opt/java/openjdk

# Configurer les variables d'environnement pour Java
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Installer les dépendances système minimales (procps est souvent requis par Spark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends procps && \
    rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances Python
# Utilisation de --no-cache-dir pour réduire la taille de l'image
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source
COPY . .

# Exposer le port de Streamlit
EXPOSE 8501

# Commande de démarrage
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
