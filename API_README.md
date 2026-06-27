# API Génération d'Images Solle (Flux + LoRA)

Ce projet déploie une API de génération d'images haute performance sur [Modal.com](https://modal.com/) en utilisant **Flux.1-schnell** et le LoRA personnalisé "Solle".

## Fonctionnalités

*   **Texte-vers-Image (Txt2Img)** : Génère une image à partir d'un prompt textuel.
*   **Image-vers-Image (Img2Img)** : Modifie une image existante selon un prompt.
*   **Scale-to-Zero** : Les serveurs GPU (NVIDIA A10G) s'éteignent après 60 secondes d'inactivité (0$ de coût pendant qu'ils sont éteints).
*   **Snapshots Mémoire** : Temps de démarrage à froid réduits.
*   **Rate Limiting** : Protection des quotas journaliers par token utilisateur.

## 🚀 Déploiement

1.  Assurez-vous que l'environnement virtuel est activé et que `modal` est configuré (`modal setup`).
2.  Assurez-vous d'avoir créé le secret des tokens d'API (déjà fait) :
    ```bash
    .\venv\Scripts\modal.exe secret create api-tokens VALID_TOKENS='["SOLLE2026"]'
    ```
3.  Déployez l'application sur Modal :
    ```bash
    .\venv\Scripts\modal.exe deploy generate_api_modal.py
    ```
4. Modal vous renverra l'URL publique de votre API.

## 📡 Utilisation de l'API

Remplacez `https://VOTRE-URL-MODAL.modal.run` par l'URL fournie lors du déploiement.

### 1. Générer une image (Text-to-Image)

*   **Endpoint** : `POST /generate`
*   **Header** : `X-API-Key: SOLLE2026`

**Exemple cURL :**
```bash
curl -X POST "https://VOTRE-URL-MODAL.modal.run/generate" \
     -H "X-API-Key: SOLLE2026" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "drawn in a comic illustration style, standing in London next to a red phone booth",
           "width": 1024,
           "height": 1024
         }' \
     --output "output_image.webp"
```
*(L'image téléchargée sera sauvegardée sous le nom `output_image.webp`)*

### 2. Générer avec Image Initiale (Image-to-Image)

Pour l'Image-to-Image (incrustation de la mascotte, ou altération d'une scène), vous devez encoder votre image de base en base64.

**Exemple en Python :**
```python
import base64
import requests

# 1. Encode l'image source en base64
with open("ma_scene_source.jpg", "rb") as image_file:
    b64_image = base64.b64encode(image_file.read()).decode("utf-8")

# 2. Prépare la requête
payload = {
    "prompt": "sollechar, riding a skateboard in this sunny park",
    "init_image": b64_image,
    "strength": 0.65, # Plus la valeur est élevée, plus l'image source est modifiée
    "width": 1024,
    "height": 1024
}

# 3. Appel de l'API
response = requests.post(
    "https://VOTRE-URL-MODAL.modal.run/generate",
    headers={"X-API-Key": "SOLLE2026", "Content-Type": "application/json"},
    json=payload
)

if response.status_code == 200:
    with open("resultat.webp", "wb") as f:
        f.write(response.content)
else:
    print("Erreur:", response.text)
```

### 3. Consulter les Statistiques de Consommation

*   **Endpoint** : `GET /stats`
*   *(Ne nécessite pas d'API Key par défaut, mais peut être restreint)*

```bash
curl "https://VOTRE-URL-MODAL.modal.run/stats"
```
**Réponse :**
```json
{
  "monthly_spend": 1.25,
  "images_today": 12,
  "avg_generation_time": "~12s",
  "remaining_budget": 28.75
}
```

## 🛠 Configurations et Quotas

*   **Timeout GPU** : Configuré à 120 secondes maximum par appel pour éviter toute surcharge de facturation en cas de plantage.
*   **Limites d'utilisation** : Fixées à 50 images par jour et par token pour protéger le budget. Vous pouvez changer cette limite dans le code (`count >= 50`).
*   **Format de Sortie** : L'API retourne nativement les images en **WEBP** avec une qualité de 80, ce qui réduit drastiquement la bande passante utilisée tout en préservant la qualité visuelle de Flux.

## Résolution

* Les largeurs (`width`) et hauteurs (`height`) passées dans la requête JSON doivent idéalement être des multiples de 64 (l'API les arrondira automatiquement au multiple inférieur si ce n'est pas le cas pour éviter des erreurs CUDA).
