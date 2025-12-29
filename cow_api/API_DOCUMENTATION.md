## Endpoints

### 1. Exploitations

#### `GET /farms`
Liste toutes les exploitations.

**Réponse:**
```json
{
  "total_farms": 2,
  "farms": [
    {"farm_id": "farm_001", "total_cows": 25, "total_embeddings": 125, "database_exists": true}
  ]
}
```

#### `GET /health`
État de l'API.

**Réponse:**
```json
{
  "api_status": "OK",
  "total_farms": 2,
  "total_cows_all_farms": 40,
  "farms": ["farm_001", "farm_002"]
}
```

---

### 2. Vaches

#### `POST /add-cow`
Ajoute une vache avec ses images.

**Content-Type:** `multipart/form-data`

**Paramètres:**
- `farm_id` (string, requis)
- `cow_id` (string, requis)
- `images` (file[], requis)

**Réponse succès (200):**
```json
{
  "message": "✅ Vache cow_001 ajoutée avec 3 images valides...",
  "farm_id": "farm_001",
  "cow_id": "cow_001",
  "images_uploaded": 5,
  "images_with_muzzle_detected": 3,
  "embeddings_extracted": 3
}
```

**Erreur (400):** Aucune image valide (museau non détecté)

---

#### `GET /farm/{farm_id}/cows`
Liste les vaches d'une exploitation.

**Réponse (200):**
```json
{
  "farm_id": "farm_001",
  "total_cows": 25,
  "cows": [
    {"cow_id": "cow_001", "index": 0, "has_embedding": true, "muzzle_files_count": 5}
  ],
  "database_status": "loaded"
}
```

**Erreur (404):** Exploitation introuvable

---

#### `DELETE /farm/{farm_id}/cow/{cow_id}`
Supprime une vache et ses données.

**Réponse (200):**
```json
{
  "message": "✅ Vache cow_001 supprimée avec succès...",
  "farm_id": "farm_001",
  "cow_id": "cow_001",
  "embedding_removed": true,
  "backup_created": true,
  "remaining_cows_in_database": 24
}
```

**Erreur (404):** Vache ou exploitation introuvable

---

### 3. Prédiction

#### `POST /predict`
Identifie une vache à partir d'une image.

**Content-Type:** `multipart/form-data`

**Paramètres:**
- `farm_id` (string, requis)
- `image` (file, requis)

**Réponse - Vache identifiée (200):**
```json
{
  "prediction": "cow_123",
  "score": 0.8756,
  "farm_id": "farm_001",
  "muzzle_saved": true,
  "total_cows_in_database": 25
}
```

**Réponse - Vache inconnue (200):**
```json
{
  "prediction": "INCONNUE",
  "score": 0.4523,
  "farm_id": "farm_001"
}
```

**Réponse - Museau non détecté (200):**
```json
{
  "prediction": "MUSEAU NON DÉTECTÉ",
  "score": 0,
  "farm_id": "farm_001",
  "muzzle_saved": false
}
```

**Réponse - Base vide (200):**
```json
{
  "prediction": "BASE DE DONNÉES VIDE",
  "score": 0.0,
  "farm_id": "farm_001",
  "message": "Aucune vache enregistrée..."
}
```

**Seuil de confiance:** score ≥ 0.6 pour identification positive

---

### 4. Images

#### `GET /farm/{farm_id}/cow/{cow_id}/raw-images`
Liste les images brutes d'une vache.

**Réponse (200):**
```json
{
  "farm_id": "farm_001",
  "cow_id": "cow_123",
  "raw_images_count": 5,
  "raw_files": ["cow_123_000_image1.jpg", "cow_123_001_image2.jpg"]
}
```

---

#### `GET /farm/{farm_id}/cow/{cow_id}/muzzle-images`
Liste les images de museaux détectés.

**Réponse (200):**
```json
{
  "farm_id": "farm_001",
  "cow_id": "cow_123",
  "muzzle_images_count": 3,
  "muzzle_files": ["muzzle_cow_123_000.jpg", "muzzle_cow_123_001.jpg"]
}
```

---

### 5. Base de Données

#### `GET /farm/{farm_id}/database/info`
Informations sur la base de données d'une exploitation.

**Réponse (200):**
```json
{
  "farm_id": "farm_001",
  "total_cows": 25,
  "cow_ids": ["cow_001", "cow_002", "..."],
  "storage_location": "data/farms/farm_001/embedding_database.json"
}
```

---

#### `POST /farm/{farm_id}/database/backup`
Crée une sauvegarde de la base de données.

**Réponse (200):**
```json
{
  "message": "Backup créé avec succès",
  "farm_id": "farm_001",
  "backup_location": "data/farms/farm_001/backups/embedding_database_backup_20251229_143025.json"
}
```

---

#### `POST /farm/{farm_id}/database/reload`
Recharge la base de données depuis le fichier.

**Réponse (200):**
```json
{
  "message": "Base de données rechargée...",
  "farm_id": "farm_001",
  "total_cows": 25
}
```

---

## Codes HTTP

- **200** - Succès
- **400** - Requête invalide (ex: aucune image, aucun museau détecté)
- **404** - Exploitation ou vache introuvable
- **500** - Erreur serveur

## Notes Importantes

1. **Exploitation auto-créée:** L'endpoint `/add-cow` crée automatiquement l'exploitation si elle n'existe pas
2. **Validation farm_id:** Tous les endpoints sauf `/add-cow` valident l'existence de l'exploitation (erreur 404 si introuvable)
3. **Formats images:** JPG, JPEG, PNG
4. **Détection museau:** Obligatoire pour générer un embedding
5. **Multiple embeddings:** Une vache peut avoir plusieurs embeddings (un par image valide)
6. **Backup automatique:** Créé avant chaque suppression de vache

## Structure de Données

```
data/farms/
  └── {farm_id}/
      ├── embedding_database.json       # Base de données des embeddings
      ├── raw_images/{cow_id}/          # Images originales
      ├── muzzle_images/{cow_id}/       # Museaux détectés
      └── prediction_results/            # Museaux des prédictions
```
