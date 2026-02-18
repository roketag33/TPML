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

# Copier et rendre exécutable le script d'entrée
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Exposer le port de Streamlit
EXPOSE 8501

# Script d'entrée : charge les données si nécessaire, puis lance Streamlit
ENTRYPOINT ["/docker-entrypoint.sh"]
