import json
import os
import logging

logger = logging.getLogger(__name__)

class LocalDatabaseManager:
    def __init__(self, db_path="data/embedding_database.json"):
        """
        Gestionnaire pour la base de données embedding stockée localement
        
        Args:
            db_path: Chemin vers le fichier JSON de la base de données
        """
        self.db_path = db_path
        
    def load_database(self):
        """Charge la base de données depuis le fichier local"""
        try:
            # Créer le dossier si nécessaire
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Vérifier si le fichier existe
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    database = json.load(f)
                logger.info(f"Base de données chargée depuis {self.db_path}")
                return database
            else:
                # Créer une nouvelle base de données vide
                logger.info("Création d'une nouvelle base de données")
                new_db = {"labels": [], "embeddings": []}
                self.save_database(new_db)
                return new_db
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la base de données: {e}")
            # Retourner une base vide en cas d'erreur
            return {"labels": [], "embeddings": []}
    
    def save_database(self, database):
        """Sauvegarde la base de données dans le fichier local"""
        try:
            # Convertir les numpy arrays en listes si nécessaire
            clean_database = self._clean_database_for_json(database)
            
            # Créer le dossier si nécessaire
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Sauvegarder dans le fichier JSON
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(clean_database, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Base de données sauvegardée dans {self.db_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la base de données: {e}")
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
    
    def get_database_info(self):
        """Retourne des informations sur la base de données"""
        try:
            if os.path.exists(self.db_path):
                file_size = os.path.getsize(self.db_path)
                database = self.load_database()
                return {
                    "exists": True,
                    "path": self.db_path,
                    "size_bytes": file_size,
                    "total_cows": len(database.get("labels", [])),
                    "total_embeddings": len(database.get("embeddings", []))
                }
            else:
                return {
                    "exists": False,
                    "path": self.db_path,
                    "size_bytes": 0,
                    "total_cows": 0,
                    "total_embeddings": 0
                }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des infos: {e}")
            return {
                "exists": False,
                "error": str(e)
            }
    
    def backup_database(self):
        """Crée une sauvegarde de la base de données"""
        try:
            if not os.path.exists(self.db_path):
                logger.warning("Aucune base de données à sauvegarder")
                return None
            
            # Créer le dossier de backup
            backup_dir = "data/backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Générer un nom de fichier avec timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"embedding_database_backup_{timestamp}.json")
            
            # Copier le fichier
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Backup créé: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du backup: {e}")
            return None


# Instance globale
db_manager = LocalDatabaseManager()

def load_database():
    """Fonction helper pour charger la base de données"""
    return db_manager.load_database()

def save_database(database):
    """Fonction helper pour sauvegarder la base de données"""
    return db_manager.save_database(database)
