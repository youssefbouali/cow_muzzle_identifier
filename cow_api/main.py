from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import numpy as np
from utils.image_utils import preprocess_image, detect_muzzle
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

# Charger la base de données locale au démarrage
database = load_database()
logging.info(f"Base de données chargée avec {len(database.get('labels', []))} vaches")

# Créer les dossiers nécessaires
os.makedirs("data/prediction_results", exist_ok=True)
os.makedirs("data/raw_images", exist_ok=True)
os.makedirs("data/muzzle_images", exist_ok=True)

@app.post("/add-cow")
async def add_cow(
    cow_id: str = Form(...),
    images: List[UploadFile] = File(...)
):
    global database
    embeddings = []

    try:
        if not images or len(images) == 0:
            return JSONResponse(status_code=400, content={
                "error": "Aucune image fournie"
            })

        logging.info(f"Traitement de {len(images)} images pour la vache {cow_id}")

        # Créer les dossiers pour cette vache
        raw_images_folder = f"data/raw_images/{cow_id}"
        muzzle_folder = f"data/muzzle_images/{cow_id}"
        os.makedirs(raw_images_folder, exist_ok=True)
        os.makedirs(muzzle_folder, exist_ok=True)

        muzzle_count = 0
        images_saved = 0
        
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

            # Sauvegarder l'image du museau
            muzzle_filename = f"muzzle_{cow_id}_{muzzle_count:03d}.jpg"
            muzzle_path = os.path.join(muzzle_folder, muzzle_filename)
            cv2.imwrite(muzzle_path, muzzle_img)
            muzzle_count += 1
            logging.info(f"Museau sauvegardé: {muzzle_path}")

            # Extraire l'embedding
            img_tensor = preprocess_image(muzzle_img)
            emb = get_embedding(img_tensor)
            embeddings.append(emb)
            logging.info(f"Embedding extrait de {image.filename}")

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
        
        # Sauvegarder localement
        save_success = save_database(database)
        
        return {
            "message": f"✅ Vache {cow_id} ajoutée avec {len(embeddings)} images valides (museau détecté).",
            "images_uploaded": len(images),
            "images_saved": images_saved,
            "images_with_muzzle_detected": len(embeddings),
            "embeddings_extracted": len(embeddings),
            "raw_images_folder": raw_images_folder,
            "muzzle_images_folder": muzzle_folder,
            "muzzle_files_count": muzzle_count,
            "database_saved": save_success
        }

    except Exception as e:
        logging.error(f"Erreur lors du traitement de la vache {cow_id}: {e}")
        return JSONResponse(status_code=500, content={
            "error": f"Erreur lors du traitement: {str(e)}"
        })



@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    global database
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
            "muzzle_saved": False
        })
    
    # Générer un nom de fichier unique avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds tronquées
    muzzle_filename = f"prediction_{timestamp}_{filename_only}"
    muzzle_save_path = os.path.join("data/prediction_results", muzzle_filename)
    
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
            "muzzle_saved": True,
            "muzzle_save_path": muzzle_save_path,
            "original_filename": filename_only,
            "message": "Aucune vache enregistrée dans la base de données. Ajoutez des vaches avec /add-cow avant de faire des prédictions.",
            "total_cows_in_database": len(database.get("labels", []))
        })

    return JSONResponse({
        "prediction": label,
        "score": float(score),
        "muzzle_saved": True,
        "muzzle_save_path": muzzle_save_path,
        "original_filename": filename_only,
        "total_cows_in_database": len(database.get("labels", []))
    })


@app.get("/cow/{cow_id}/raw-images")
async def get_cow_raw_images(cow_id: str):
    """Récupère la liste des images brutes d'une vache stockées localement"""
    raw_folder = f"data/raw_images/{cow_id}"
    
    if not os.path.exists(raw_folder):
        return JSONResponse(
            status_code=404,
            content={"error": f"Aucune image brute trouvée pour la vache {cow_id}"}
        )
    
    try:
        raw_files = [f for f in os.listdir(raw_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        raw_files.sort()
        
        return {
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


@app.get("/cow/{cow_id}/muzzle-images")
async def get_cow_muzzle_images(cow_id: str):
    """Récupère la liste des images de museaux sauvegardées localement"""
    muzzle_folder = f"data/muzzle_images/{cow_id}"
    
    if not os.path.exists(muzzle_folder):
        return JSONResponse(
            status_code=404,
            content={"error": f"Aucune image de museau trouvée pour la vache {cow_id}"}
        )
    
    try:
        muzzle_files = [f for f in os.listdir(muzzle_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        muzzle_files.sort()  # Tri par nom
        
        return {
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


@app.delete("/cow/{cow_id}")
async def delete_cow(cow_id: str):
    """Supprime une vache et toutes ses images de la base de données d'embeddings"""
    global database
    
    try:
        # Vérifier si la vache existe dans la base de données
        labels = database.get("labels", [])
        embeddings = database.get("embeddings", [])
        
        if cow_id not in labels:
            return JSONResponse(
                status_code=404,
                content={"error": f"Vache {cow_id} non trouvée dans la base de données"}
            )
        
        # Trouver l'index de la vache dans la base de données
        cow_index = labels.index(cow_id)
        
        # Créer une sauvegarde avant suppression
        backup_path = db_manager.backup_database()
        if not backup_path:
            logging.warning("Impossible de créer une sauvegarde avant suppression")
        
        # Supprimer la vache et son embedding de la base de données
        labels.pop(cow_index)
        embeddings.pop(cow_index)
        
        # Mettre à jour la base de données globale
        database["labels"] = labels
        database["embeddings"] = embeddings
        
        # Sauvegarder la base de données mise à jour
        save_success = save_database(database)
        
        # Supprimer le dossier local des images brutes
        raw_folder = f"data/raw_images/{cow_id}"
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
        muzzle_folder = f"data/muzzle_images/{cow_id}"
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
            "message": f"✅ Vache {cow_id} supprimée avec succès",
            "cow_id": cow_id,
            "embedding_removed": True,
            "database_saved": save_success,
            "backup_created": backup_path is not None,
            "backup_location": backup_path if backup_path else None,
            "raw_folder_deleted": not os.path.exists(raw_folder),
            "raw_files_deleted": raw_files_deleted,
            "muzzle_folder_deleted": not os.path.exists(muzzle_folder),
            "muzzle_files_deleted": muzzle_files_deleted,
            "remaining_cows_in_database": len(database.get("labels", []))
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


@app.get("/cows")
async def list_all_cows():
    """Liste toutes les vaches présentes dans la base de données d'embeddings"""
    try:
        labels = database.get("labels", [])
        embeddings = database.get("embeddings", [])
        
        cows_info = []
        for i, cow_id in enumerate(labels):
            # Vérifier si le dossier de museaux existe localement
            muzzle_folder = f"data/muzzle_images/{cow_id}"
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
            "total_cows": len(labels),
            "cows": cows_info,
            "database_status": "loaded" if len(labels) > 0 else "empty"
        }
        
    except Exception as e:
        logging.error(f"Erreur lors de la récupération de la liste des vaches: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors de la récupération: {str(e)}"}
        )


@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    # Informations sur la base de données
    db_info = db_manager.get_database_info()
    
    return {
        "api_status": "OK",
        "database_loaded": len(database.get("labels", [])) > 0,
        "database_info": db_info,
        "total_cows_in_database": len(database.get("labels", []))
    }


@app.get("/database/info")
async def get_database_info():
    """Informations détaillées sur la base de données"""
    db_info = db_manager.get_database_info()
    
    return {
        "total_cows": len(database.get("labels", [])),
        "cow_ids": database.get("labels", []),
        "storage_location": db_manager.db_path,
        "database_details": db_info
    }


@app.post("/database/backup")
async def create_database_backup():
    """Créer une sauvegarde manuelle de la base de données"""
    backup_path = db_manager.backup_database()
    if backup_path:
        return {
            "message": "Backup créé avec succès",
            "backup_location": backup_path
        }
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Échec de la création du backup"}
        )


@app.post("/database/reload")
async def reload_database():
    """Recharge la base de données depuis le fichier local"""
    global database
    try:
        database = load_database()
        return {
            "message": "Base de données rechargée depuis le fichier local",
            "total_cows": len(database.get("labels", [])),
            "cow_ids": database.get("labels", [])
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Erreur lors du rechargement: {str(e)}"}
        )


if __name__ == "__main__":
        import uvicorn
        uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
