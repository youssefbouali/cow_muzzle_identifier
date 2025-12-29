import json
import os
import logging

logger = logging.getLogger(__name__)

class LocalDatabaseManager:
    def __init__(self, base_path="data/farms"):
        """
        Gestionnaire pour les bases de données embedding par exploitation
        
        Args:
            base_path: Chemin de base vers le dossier contenant les exploitations
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def get_farm_db_path(self, farm_id):
        """Retourne le chemin de la base de données pour une exploitation"""
        return os.path.join(self.base_path, farm_id, "embedding_database.json")
        
    def load_database(self, farm_id):
        """Charge la base de données d'une exploitation spécifique"""
        try:
            db_path = self.get_farm_db_path(farm_id)
            
            # Créer le dossier si nécessaire
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Vérifier si le fichier existe
            if os.path.exists(db_path):
                with open(db_path, 'r', encoding='utf-8') as f:
                    database = json.load(f)
                logger.info(f"Base de données chargée depuis {db_path}")
                return database
            else:
                # Créer une nouvelle base de données vide
                logger.info(f"Création d'une nouvelle base de données pour l'exploitation {farm_id}")
                new_db = {"labels": [], "embeddings": []}
                self.save_database(new_db, farm_id)
                return new_db
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la base de données pour {farm_id}: {e}")
            # Retourner une base vide en cas d'erreur
            return {"labels": [], "embeddings": []}
    
    def save_database(self, database, farm_id):
        """Sauvegarde la base de données pour une exploitation spécifique"""
        try:
            db_path = self.get_farm_db_path(farm_id)
            
            # Convertir les numpy arrays en listes si nécessaire
            clean_database = self._clean_database_for_json(database)
            
            # Créer le dossier si nécessaire
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            # Sauvegarder dans le fichier JSON
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(clean_database, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Base de données sauvegardée dans {db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la base de données pour {farm_id}: {e}")
            return False
    
    def _clean_database_for_json(self, database):
        """Nettoie la base de données pour la rendre compatible JSON"""
        import numpy as np
        
        clean_db = {
            "labels": database.get("labels", []),
            "embeddings": []
        }
        
        for emb in database.get("embeddings", []):
            if isinstance(emb, np.ndarray):
                clean_db["embeddings"].append(emb.tolist())
            else:
                clean_db["embeddings"].append(emb)
        
        return clean_db
    
    def get_database_info(self, farm_id):
        """Retourne des informations sur la base de données d'une exploitation"""
        try:
            db_path = self.get_farm_db_path(farm_id)
            
            if os.path.exists(db_path):
                file_size = os.path.getsize(db_path)
                database = self.load_database(farm_id)
                return {
                    "exists": True,
                    "path": db_path,
                    "farm_id": farm_id,
                    "size_bytes": file_size,
                    "total_cows": len(database.get("labels", [])),
                    "total_embeddings": len(database.get("embeddings", []))
                }
            else:
                return {
                    "exists": False,
                    "path": db_path,
                    "farm_id": farm_id,
                    "size_bytes": 0,
                    "total_cows": 0,
                    "total_embeddings": 0
                }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos pour {farm_id}: {e}")
            return {
                "exists": False,
                "farm_id": farm_id,
                "error": str(e)
            }
    
    def list_all_farms(self):
        """Liste toutes les exploitations enregistrées"""
        try:
            if not os.path.exists(self.base_path):
                return []
            
            farms = []
            for item in os.listdir(self.base_path):
                item_path = os.path.join(self.base_path, item)
                if os.path.isdir(item_path):
                    db_path = self.get_farm_db_path(item)
                    if os.path.exists(db_path):
                        farms.append(item)
            
            return farms
        except Exception as e:
            logger.error(f"Erreur lors de la liste des exploitations: {e}")
            return []
    
    def farm_exists(self, farm_id):
        """Vérifie si une exploitation existe"""
        db_path = self.get_farm_db_path(farm_id)
        return os.path.exists(db_path)
    
    def backup_database(self, farm_id):
        """Crée une sauvegarde de la base de données d'une exploitation"""
        try:
            db_path = self.get_farm_db_path(farm_id)
            
            if not os.path.exists(db_path):
                logger.warning(f"Aucune base de données à sauvegarder pour {farm_id}")
                return None
            
            # Créer le dossier de backup
            backup_dir = os.path.join(self.base_path, farm_id, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Générer un nom de fichier avec timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"embedding_database_backup_{timestamp}.json")
            
            # Copier le fichier
            import shutil
            shutil.copy2(db_path, backup_path)
            
            logger.info(f"Backup créé: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du backup pour {farm_id}: {e}")
            return None


# Instance globale
db_manager = LocalDatabaseManager()

def load_database(farm_id):
    """Fonction helper pour charger la base de données d'une exploitation"""
    return db_manager.load_database(farm_id)

def save_database(database, farm_id):
    """Fonction helper pour sauvegarder la base de données d'une exploitation"""
    return db_manager.save_database(database, farm_id)
