# ☁️ Guide de Déploiement Automatisé (CI/CD)

Pour que **Watchtower** mette à jour votre application automatiquement à chaque `git push`, nous utilisons **GitHub Actions**.

## 1. Configuration GitHub (CI/CD)

Le fichier `.github/workflows/docker-publish.yml` a été créé. Il va automatiquement :
1.  Construire votre image Docker.
2.  La publier sur le **GitHub Container Registry (GHCR)**.

### Actions requises de votre part :
1.  Poussez le code sur GitHub : `git add . && git commit -m "Add CI/CD" && git push`
2.  Allez dans l'onglet **Actions** de votre dépôt GitHub pour vérifier que le build "Docker Image CI" passe au vert.
3.  Une fois terminé, votre image sera disponible à l'adresse : `ghcr.io/VOTRE_USERNAME/NOM_DU_REPO:latest`.

## 2. Configuration Portainer (VPS)

1.  Ouvrez le fichier `docker-compose.vps.yml` sur votre machine.
2.  **Modifiez la ligne 8** :
    ```yaml
    image: ghcr.io/votre-user/tpml:latest
    ```
    Remplacez `votre-user` par votre nom d'utilisateur GitHub (en minuscule) et `tpml` par le nom de votre dépôt.

3.  Dans Portainer > **Stacks** > **Add stack** :
    *   Copiez-collez le contenu de votre `docker-compose.vps.yml` (modifié).
    *   **Authentification (Important)** : GHCR est privé par défaut. Si votre repo est privé, vous devez configurer un "Registry" dans Portainer avec vos identifiants GitHub (générez un Personal Access Token avec les droits `read:packages`).

## 3. Le cycle de vie "Magic" ✨

1.  Vous modifiez votre code localement.
2.  Vous faites un `git push`.
3.  **GitHub Actions** construit la nouvelle image et la pousse sur GHCR.
4.  Sur votre VPS, **Watchtower** (qui tourne déjà grâce au docker-compose) détecte la nouvelle image dans les 5 minutes.
5.  Watchtower télécharge la nouvelle image et redémarre le conteneur `tpml-app` avec la nouvelle version.

🚀 **Zéro intervention manuelle sur le serveur !**
