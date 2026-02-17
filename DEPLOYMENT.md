# ☁️ Guide de Déploiement VPS avec Portainer

Ce guide explique comment déployer le projet sur votre VPS en utilisant **Portainer** et **Watchtower** pour les mises à jour automatiques.

## Prérequis
- Un VPS avec Docker et Portainer installés.
- Ce projet hébergé sur un dépôt Git accessible (GitHub, GitLab, etc.).

## Méthode Recommandée : Portainer "Stacks" (Git Repository)

Cette méthode permet à Portainer de construire l'image directement depuis votre code source.

1.  Connectez-vous à votre **Portainer**.
2.  Allez dans **Stacks** > **+ Add stack**.
3.  Sélectionnez **Repository**.
4.  Remplissez les champs :
    *   **Name** : `tpml-stack` (par exemple).
    *   **Repository URL** : L'URL HTTPS de votre dépôt Git (ex: `https://github.com/votre-user/TPML.git`).
    *   **Repository reference** : `main` (ou la branche que vous utilisez).
    *   **Compose path** : `docker-compose.vps.yml` (⚠️ Important : on utilise le fichier spécifique VPS créé).
    *   **Automatic updates** : Activez cette option ("Webhook" ou "Polling").
        *   *Note : C'est ici que Portainer va mettre à jour votre CODE application. Watchtower, lui, mettra à jour les IMAGES de base (Mongo, Redis, Cassandra).*

5.  Cliquez sur **Deploy the stack**.

## Rôle de Watchtower

Le fichier `docker-compose.vps.yml` inclut un service **Watchtower**. Son rôle est le suivant :

*   Il surveille les conteneurs en cours d'exécution.
*   Si une nouvelle version des images de base (ex: `mongo:latest`, `redis:latest`) est publiée sur le Docker Hub, Watchtower va :
    1.  Télécharger la nouvelle image.
    2.  Arrêter le conteneur proprement.
    3.  Le relancer avec la nouvelle image et les mêmes options.
*   Il nettoie automatiquement les vieilles images (`WATCHTOWER_CLEANUP=true`).

## Initialisation du Replica Set (Premier Lancement)

Comme sur votre machine locale, le Replica Set MongoDB doit être initialisé une seule fois au premier démarrage.

1.  Une fois la stack déployée ("Running"), ouvrez la console du conteneur `tpml-mongo1` via Portainer :
    *   Cliquez sur le conteneur `tpml-mongo1`.
    *   Cliquez sur **Console** > **Connect**.
2.  Lancez le script d'initialisation (si vous avez monté le volume du code) OU utilisez la commande manuelle suivante directement dans la console :

```bash
mongosh --eval 'rs.initiate({_id: "rs0", members: [{_id: 0, host: "mongo1:27017"}, {_id: 1, host: "mongo2:27017"}, {_id: 2, host: "mongo3:27017"}]})'
```

Si la commande renvoie `{ "ok" : 1 }`, votre cluster est opérationnel !
