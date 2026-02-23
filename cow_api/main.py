from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import numpy as np
from utils.image_utils import preprocess_image, detect_muzzle, generate_augmentations
from utils.embeddings import get_embedding, predict_identity
from utils.local_database import db_manager, load_database, save_database
import cv2
import logging
from dotenv import load_dotenv
from datetime import datetime
from typing import List

# Charger les variables d'environnement
load_dotenv()

# Configuration des logs
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Configuration CORS
app.add_middleware( 
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache des bases de données par exploitation
databases_cache = {}

def validate_farm_exists(farm_id: str):
    """Valide qu'une exploitation existe, retourne une réponse d'erreur si elle n'existe pas"""
    if not db_manager.farm_exists(farm_id):
        return JSONResponse(
            status_code=404,
            content={
                "error": f"Exploitation '{farm_id}' introuvable",
                "farm_id": farm_id,
                "message": "Cette exploitation n'existe pas. Utilisez GET /farms pour voir les exploitations disponibles ou ajoutez une vache avec POST /add-cow pour créer une nouvelle exploitation."
            }
        )
    return None

def get_farm_database(farm_id: str):
    """Charge ou récupère la base de données d'une exploitation depuis le cache"""
    if farm_id not in databases_cache:
        databases_cache[farm_id] = load_database(farm_id)
        logging.info(f"Base de données chargée pour l'exploitation {farm_id}: {len(databases_cache[farm_id].get('labels', []))} vaches")
    return databases_cache[farm_id]

def get_farm_folders(farm_id: str):
    """Retourne les chemins des dossiers pour une exploitation"""
    return {
        "base": f"data/farms/{farm_id}",
        "prediction_results": f"data/farms/{farm_id}/prediction_results",
        "raw_images": f"data/farms/{farm_id}/raw_images",
        "muzzle_images": f"data/farms/{farm_id}/muzzle_images"
    }

# Créer le dossier de base pour les exploitations
os.makedirs("data/farms", exist_ok=True)

@app.post("/add-cow")
async def add_cow(
    farm_id: str = Form(...),
    cow_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    embeddings = []

    try:
        if not images or len(images) == 0:
            return JSONResponse(status_code=400, content={
                "error": "Aucune image fournie"
            })

        logging.info(f"Traitement de {len(images)} images pour la vache {cow_id} de l'exploitation {farm_id}")

        # Récupérer la base de données de l'exploitation
        database = get_farm_database(farm_id)
        folders = get_farm_folders(farm_id)
        
        # Créer les dossiers pour cette vache
        raw_images_folder = os.path.join(folders["raw_images"], cow_id)
        muzzle_folder = os.path.join(folders["muzzle_images"], cow_id)
        os.makedirs(raw_images_folder, exist_ok=True)
        os.makedirs(muzzle_folder, exist_ok=True)
        os.makedirs(folders["prediction_results"], exist_ok=True)

        muzzle_count = 0
        images_saved = 0
        total_embeddings_created = 0
        
        for i, image in enumerate(images):
            # Sauvegarder l'image brute
            raw_image_path = os.path.join(raw_images_folder, f"{cow_id}_{i:03d}_{image.filename}")
            with open(raw_image_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            images_saved += 1
            logging.info(f"Image brute sauvegardée: {raw_image_path}")

            # Détecter le museau
            muzzle_img = detect_muzzle(raw_image_path)
            if muzzle_img is None:
                logging.info(f"Museau non détecté dans l'image {image.filename}")
                continue

            # Générer des augmentations (Original, Flip, Gauche, Droite)
            augmented_images = generate_augmentations(muzzle_img)
            
            aug_labels = ["original", "flipped", "left_view", "right_view"]
            
            for idx, aug_img in enumerate(augmented_images):
                # Sauvegarder l'image du museau (originale ou augmentée)
                aug_type = aug_labels[idx] if idx < len(aug_labels) else f"aug_{idx}"
                muzzle_filename = f"muzzle_{cow_id}_{muzzle_count:03d}_{aug_type}.jpg"
                muzzle_path = os.path.join(muzzle_folder, muzzle_filename)
                cv2.imwrite(muzzle_path, aug_img)
                
                # Extraire l'embedding
                img_tensor = preprocess_image(aug_img)
                emb = get_embedding(img_tensor)
                embeddings.append(emb)
                
                muzzle_count += 1
                total_embeddings_created += 1
                logging.info(f"Embedding extrait pour augmentation: {aug_type}")

        if len(embeddings) == 0:
            return JSONResponse(status_code=400, content={
                "error": "Aucune image valide (museau non détecté) trouvée.",
                "images_uploaded": len(images),
                "images_saved": images_saved
            })

        # Sauvegarder chaque embedding individuellement avec l'ID de la vache
        for emb in embeddings:
            database["labels"].append(cow_id)
            database["embeddings"].append(emb.tolist())
        
        # Mettre à jour le cache
        databases_cache[farm_id] = database
        
        # Sauvegarder localement
        save_success = save_database(database, farm_id)
        
        return {
            "message": f"✅ Vache {cow_id} ajoutée avec {total_embeddings_created} signatures (incluant augmentations 3D) pour {images_saved} images traitées.",
            "farm_id": farm_id,
            "cow_id": cow_id,
            "images_uploaded": len(images),
            "images_saved": images_saved,
            "total_embeddings": total_embeddings_created,
            "augmentations_per_image": len(augmented_images) if 'augmented_images' in locals() else 0,
            "raw_images_folder": raw_images_folder,
            "muzzle_images_folder": muzzle_folder,
            "database_saved": save_success
        }

    except Exception as e:
        logging.error(f"Erreur lors du traitement de la vache {cow_id} pour l'exploitation {farm_id}: {e}")
        return JSONResponse(status_code=500, content={
            "error": f"Erreur lors du traitement: {str(e)}",
            "farm_id": farm_id,
            "cow_id": cow_id
        })



@app.post("/predict")
async def predict(
    farm_id: str = Form(...),
    image: UploadFile = File(...)
):
    # Valider que l'exploitation existe
    farm_error = validate_farm_exists(farm_id)
    if farm_error:
        return farm_error
    
    filename_only = os.path.basename(image.filename)
    temp_path = f"temp_{filename_only}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Détection du museau
    muzzle_img = detect_muzzle(temp_path)

    os.remove(temp_path)

    if muzzle_img is None:
        return JSONResponse({
            "prediction": "MUSEAU NON DÉTECTÉ",
            "score": 0,
            "farm_id": farm_id,
            "muzzle_saved": False
        })
    
    # Récupérer la base de données de l'exploitation
    database = get_farm_database(farm_id)
    folders = get_farm_folders(farm_id)
    
    # Générer un nom de fichier unique avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds tronquées
    muzzle_filename = f"prediction_{timestamp}_{filename_only}"
    muzzle_save_path = os.path.join(folders["prediction_results"], muzzle_filename)
    os.makedirs(folders["prediction_results"], exist_ok=True)
    
    # Sauvegarder l'image du museau détecté
    cv2.imwrite(muzzle_save_path, muzzle_img)
    logging.info(f"Museau détecté sauvegardé: {muzzle_save_path}")
    
    img_tensor = preprocess_image(muzzle_img)
    label, score = predict_identity(img_tensor, database, threshold=0.6)

    # Gestion du cas où la base de données est vide
    if label == "BASE_VIDE":
        return JSONResponse({
            "prediction": "BASE DE DONNÉES VIDE",
            "score": 0.0,
            "farm_id": farm_id,
            "muzzle_saved": True,
            "muzzle_save_path": muzzle_save_path,
            "original_filename": filename_only,
            "message": f"Aucune vache enregistrée dans la base de données de l'exploitation {farm_id}. Ajoutez des vaches avec /add-cow avant de faire des prédictions.",
            "total_cows_in_database": len(database.get("labels", []))
        })

    return JSONResponse({
        "prediction": label,
        "score": float(score),
        "farm_id": farm_id,
        "muzzle_saved": True,
        "muzzle_save_path": muzzle_save_path,
        "original_filename": filename_only,
        "total_cows_in_database": len(database.get("labels", []))
    })


@app.get("/farm/{farm_id}/cow/{cow_id}/raw-images")
async def get_cow_raw_images(farm_id: str, cow_id: str):
    """Récupère la liste des images brutes d'une vache stockées localement"""
    # Valider que l'exploitation existe
    farm_error = validate_farm_exists(farm_id)
    if farm_error:
        return farm_error
    
    folders = get_farm_folders(farm_id)
    raw_folder = os.path.join(folders["raw_images"], cow_id)
    
    if not os.path.exists(raw_folder):
        return JSONResponse(
            status_code=404,
            content={"error": f"Aucune image brute trouvée pour la vache {cow_id} de l'exploitation {farm_id}"}
        )
    
    try:
        raw_files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        raw_files.sort()
        
        return {
            "farm_id": farm_id,
            "cow_id": cow_id,
            "raw_images_count": len(raw_files),
            "raw_folder": raw_folder,
            "raw_files": raw_files
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors de la lecture du dossier: {str(e)}"}
        )


@app.get("/farm/{farm_id}/cow/{cow_id}/muzzle-images")
async def get_cow_muzzle_images(farm_id: str, cow_id: str):
    """Récupère la liste des images de museaux sauvegardées localement"""
    # Valider que l'exploitation existe
    farm_error = validate_farm_exists(farm_id)
    if farm_error:
        return farm_error
    
    folders = get_farm_folders(farm_id)
    muzzle_folder = os.path.join(folders["muzzle_images"], cow_id)
    
    if not os.path.exists(muzzle_folder):
        return JSONResponse(
            status_code=404,
            content={"error": f"Aucune image de museau trouvée pour la vache {cow_id} de l'exploitation {farm_id}"}
        )
    
    try:
        muzzle_files = [f for f in os.listdir(muzzle_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        muzzle_files.sort()  # Tri par nom
        
        return {
            "farm_id": farm_id,
            "cow_id": cow_id,
            "muzzle_images_count": len(muzzle_files),
            "muzzle_folder": muzzle_folder,
            "muzzle_files": muzzle_files
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors de la lecture du dossier: {str(e)}"}
        )


@app.delete("/farm/{farm_id}/cow/{cow_id}")
async def delete_cow(farm_id: str, cow_id: str):
    """Supprime une vache et toutes ses images de la base de données d'embeddings d'une exploitation"""
    # Valider que l'exploitation existe
    farm_error = validate_farm_exists(farm_id)
    if farm_error:
        return farm_error
    
    try:
        # Récupérer la base de données de l'exploitation
        database = get_farm_database(farm_id)
        folders = get_farm_folders(farm_id)
        
        # Vérifier si la vache existe dans la base de données
        labels = database.get("labels", [])
        embeddings = database.get("embeddings", [])
        
        if cow_id not in labels:
            return JSONResponse(
                status_code=404,
                content={"error": f"Vache {cow_id} non trouvée dans la base de données de l'exploitation {farm_id}"}
            )
        
        # Compter le nombre d'embeddings de cette vache
        embeddings_count = labels.count(cow_id)
        logging.info(f"Suppression de {embeddings_count} embedding(s) pour la vache {cow_id}")
        
        # Créer une sauvegarde avant suppression
        backup_path = db_manager.backup_database(farm_id)
        if not backup_path:
            logging.warning("Impossible de créer une sauvegarde avant suppression")
        
        # Supprimer TOUS les embeddings de cette vache
        # Créer de nouvelles listes sans les entrées de cette vache
        new_labels = []
        new_embeddings = []
        for i, label in enumerate(labels):
            if label != cow_id:
                new_labels.append(label)
                new_embeddings.append(embeddings[i])
        
        # Mettre à jour la base de données
        database["labels"] = new_labels
        database["embeddings"] = new_embeddings
        
        # Mettre à jour le cache
        databases_cache[farm_id] = database
        
        # Sauvegarder la base de données mise à jour
        save_success = save_database(database, farm_id)
        
        # Supprimer le dossier local des images brutes
        raw_folder = os.path.join(folders["raw_images"], cow_id)
        raw_files_deleted = 0
        if os.path.exists(raw_folder):
            try:
                raw_files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                raw_files_deleted = len(raw_files)
                shutil.rmtree(raw_folder)
                logging.info(f"Dossier d'images brutes {raw_folder} supprimé avec {raw_files_deleted} fichiers")
            except Exception as e:
                logging.warning(f"Impossible de supprimer le dossier {raw_folder}: {e}")
        
        # Supprimer le dossier local des images de museaux
        muzzle_folder = os.path.join(folders["muzzle_images"], cow_id)
        muzzle_files_deleted = 0
        if os.path.exists(muzzle_folder):
            try:
                muzzle_files = [f for f in os.listdir(muzzle_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                muzzle_files_deleted = len(muzzle_files)
                shutil.rmtree(muzzle_folder)
                logging.info(f"Dossier de museaux {muzzle_folder} supprimé avec {muzzle_files_deleted} fichiers")
            except Exception as e:
                logging.warning(f"Impossible de supprimer le dossier {muzzle_folder}: {e}")
        
        return {
            "message": f"✅ Vache {cow_id} supprimée avec succès de l'exploitation {farm_id}",
            "farm_id": farm_id,
            "cow_id": cow_id,
            "embeddings_removed": embeddings_count,
            "database_saved": save_success,
            "backup_created": backup_path is not None,
            "backup_location": backup_path if backup_path else None,
            "raw_folder_deleted": not os.path.exists(raw_folder),
            "raw_files_deleted": raw_files_deleted,
            "muzzle_folder_deleted": not os.path.exists(muzzle_folder),
            "muzzle_files_deleted": muzzle_files_deleted,
            "remaining_cows_in_database": len(set(database.get("labels", [])))
        }
        
    except Exception as e:
        logging.error(f"Erreur lors de la suppression de la vache {cow_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Erreur lors de la suppression: {str(e)}",
                "cow_id": cow_id
            }
        )


@app.get("/farm/{farm_id}/cows")
async def list_all_cows(farm_id: str):
    """Liste toutes les vaches présentes dans la base de données d'embeddings d'une exploitation"""
    # Valider que l'exploitation existe
    farm_error = validate_farm_exists(farm_id)
    if farm_error:
        return farm_error
    
    try:
        database = get_farm_database(farm_id)
        folders = get_farm_folders(farm_id)
        
        labels = database.get("labels", [])
        embeddings = database.get("embeddings", [])
        
        cows_info = []
        for i, cow_id in enumerate(labels):
            # Vérifier si le dossier de museaux existe localement
            muzzle_folder = os.path.join(folders["muzzle_images"], cow_id)
            muzzle_files_count = 0
            if os.path.exists(muzzle_folder):
                muzzle_files = [f for f in os.listdir(muzzle_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                muzzle_files_count = len(muzzle_files)
            
            cows_info.append({
                "cow_id": cow_id,
                "index": i,
                "has_embedding": i < len(embeddings),
                "muzzle_folder_exists": os.path.exists(muzzle_folder),
                "muzzle_files_count": muzzle_files_count
            })
        
        return {
            "farm_id": farm_id,
            "total_cows": len(labels),
            "cows": cows_info,
            "database_status": "loaded" if len(labels) > 0 else "empty"
        }
        
    except Exception as e:
        logging.error(f"Erreur lors de la récupération de la liste des vaches pour {farm_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors de la récupération: {str(e)}", "farm_id": farm_id}
        )


@app.get("/farms")
async def list_all_farms():
    """Liste toutes les exploitations enregistrées"""
    try:
        farms = db_manager.list_all_farms()
        
        farms_info = []
        for farm_id in farms:
            db_info = db_manager.get_database_info(farm_id)
            farms_info.append({
                "farm_id": farm_id,
                "total_cows": db_info.get("total_cows", 0),
                "total_embeddings": db_info.get("total_embeddings", 0),
                "database_exists": db_info.get("exists", False)
            })
        
        return {
            "total_farms": len(farms),
            "farms": farms_info
        }
    except Exception as e:
        logging.error(f"Erreur lors de la récupération de la liste des exploitations: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors de la récupération: {str(e)}"}
        )


@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    farms = db_manager.list_all_farms()
    total_cows = 0
    
    for farm_id in farms:
        db = get_farm_database(farm_id)
        total_cows += len(db.get("labels", []))
    
    return {
        "api_status": "OK",
        "total_farms": len(farms),
        "total_cows_all_farms": total_cows,
        "farms": farms
    }


@app.get("/farm/{farm_id}/database/info")
async def get_database_info(farm_id: str):
    """Informations détaillées sur la base de données d'une exploitation"""
    # Valider que l'exploitation existe
    farm_error = validate_farm_exists(farm_id)
    if farm_error:
        return farm_error
    
    database = get_farm_database(farm_id)
    db_info = db_manager.get_database_info(farm_id)
    
    return {
        "farm_id": farm_id,
        "total_cows": len(database.get("labels", [])),
        "cow_ids": database.get("labels", []),
        "storage_location": db_manager.get_farm_db_path(farm_id),
        "database_details": db_info
    }


@app.post("/farm/{farm_id}/database/backup")
async def create_database_backup(farm_id: str):
    """Créer une sauvegarde manuelle de la base de données d'une exploitation"""
    # Valider que l'exploitation existe
    farm_error = validate_farm_exists(farm_id)
    if farm_error:
        return farm_error
    
    backup_path = db_manager.backup_database(farm_id)
    if backup_path:
        return {
            "message": "Backup créé avec succès",
            "farm_id": farm_id,
            "backup_location": backup_path
        }
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Échec de la création du backup", "farm_id": farm_id}
        )


@app.post("/farm/{farm_id}/database/reload")
async def reload_database(farm_id: str):
    """Recharge la base de données d'une exploitation depuis le fichier local"""
    # Valider que l'exploitation existe
    farm_error = validate_farm_exists(farm_id)
    if farm_error:
        return farm_error
    
    try:
        database = load_database(farm_id)
        databases_cache[farm_id] = database
        return {
            "message": f"Base de données rechargée depuis le fichier local pour l'exploitation {farm_id}",
            "farm_id": farm_id,
            "total_cows": len(database.get("labels", [])),
            "cow_ids": database.get("labels", [])
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors du rechargement: {str(e)}", "farm_id": farm_id}
        )


if __name__ == "__main__":
        import uvicorn
        uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
