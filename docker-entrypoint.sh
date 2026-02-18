#!/bin/bash
set -e

echo "========================================="
echo "  TPML - Initialisation du conteneur"
echo "========================================="

# Fonction pour attendre qu'un service soit prêt
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_retries=30
    local retry=0

    echo "⏳ Attente de $service_name ($host:$port)..."
    while ! python -c "import socket; s=socket.socket(); s.settimeout(2); s.connect(('$host', $port)); s.close()" 2>/dev/null; do
        retry=$((retry + 1))
        if [ $retry -ge $max_retries ]; then
            echo "❌ $service_name n'est pas disponible après $max_retries tentatives."
            break
        fi
        echo "  Tentative $retry/$max_retries..."
        sleep 2
    done
    echo "✅ $service_name est prêt !"
}

# Attendre que les BDD soient prêtes
MONGO_HOST="${MONGO_URI:-localhost}"
REDIS_HOST_VAR="${REDIS_HOST:-localhost}"
CASSANDRA_HOST_VAR="${CASSANDRA_HOST:-localhost}"

wait_for_service "mongo1" 27017 "MongoDB"
wait_for_service "$REDIS_HOST_VAR" 6379 "Redis"
wait_for_service "$CASSANDRA_HOST_VAR" 9042 "Cassandra"

# Charger les données si MongoDB est vide
echo ""
echo "📦 Vérification et chargement des données..."
python -c "
import os
from pymongo import MongoClient

mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
db = client['tpml_iris']
count = db['iris_data'].count_documents({})

if count == 0:
    print('📥 Base vide, chargement des données...')
    import subprocess
    subprocess.run(['python', 'src/data_loader.py'], check=True)
    subprocess.run(['python', 'src/create_indexes.py'], check=True)
    print('✅ Données chargées et index créés !')
else:
    print(f'✅ {count} documents déjà présents, pas de rechargement.')
"

echo ""
echo "🚀 Démarrage de Streamlit..."
exec streamlit run src/app.py --server.address=0.0.0.0
