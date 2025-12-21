import json
import boto3
import os
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

class S3DatabaseManager:
    def __init__(self, bucket_name=None, region_name='eu-north-1'):
        """
        Gestionnaire pour la base de données embedding stockée sur S3
        """
        self.bucket_name = bucket_name or os.getenv('AWS_S3_BUCKET', 'boviclouds-cows-imgs')
        self.region_name = region_name
        self.db_key = "database/embedding_database.json"
        self.local_cache = "utils/embedding_database_cache.json"
        
        # Initialisation du client S3
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region_name,
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du client S3: {e}")
            raise
    
    def load_database(self):
        """Charge la base de données depuis S3 avec cache local"""
        try:
            # Essayer de charger depuis S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=self.db_key
            )
            content = response['Body'].read().decode('utf-8')
            database = json.loads(content)
            
            # Sauvegarder en cache local
            os.makedirs(os.path.dirname(self.local_cache), exist_ok=True)
            with open(self.local_cache, 'w') as f:
                json.dump(database, f, indent=2)
            
            logger.info(f"Base de données chargée depuis S3: {self.db_key}")
            return database
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                # Base de données n'existe pas encore, créer une nouvelle
                logger.info("Création d'une nouvelle base de données")
                new_db = {"labels": [], "embeddings": []}
                self.save_database(new_db)
                return new_db
            else:
                # Erreur S3, essayer le cache local
                logger.warning(f"Erreur S3: {e}, utilisation du cache local")
                return self._load_local_cache()
        except Exception as e:
            logger.error(f"Erreur lors du chargement depuis S3: {e}")
            return self._load_local_cache()
    
    def save_database(self, database):
        """Sauvegarde la base de données sur S3 et localement"""
        try:
            # Convertir les numpy arrays en listes si nécessaire
            clean_database = self._clean_database_for_json(database)
            
            # Sauvegarder sur S3
            json_content = json.dumps(clean_database, indent=2)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.db_key,
                Body=json_content,
                ContentType='application/json'
            )
            
            # Sauvegarder en cache local
            os.makedirs(os.path.dirname(self.local_cache), exist_ok=True)
            with open(self.local_cache, 'w') as f:
                json.dump(clean_database, f, indent=2)
            
            logger.info(f"Base de données sauvegardée sur S3: {self.db_key}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde S3: {e}")
            # Au moins sauvegarder localement
            try:
                clean_database = self._clean_database_for_json(database)
                os.makedirs(os.path.dirname(self.local_cache), exist_ok=True)
                with open(self.local_cache, 'w') as f:
                    json.dump(clean_database, f, indent=2)
                logger.info("Sauvegarde locale de secours effectuée")
            except Exception as local_error:
                logger.error(f"Échec de la sauvegarde locale: {local_error}")
            return False
    
    def _clean_database_for_json(self, database):
        """Nettoie la base de données pour la serialisation JSON"""
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
    
    def _load_local_cache(self):
        """Charge le cache local en cas de problème S3"""
        try:
            if os.path.exists(self.local_cache):
                with open(self.local_cache, 'r') as f:
                    data = json.load(f)
                logger.info("Base de données chargée depuis le cache local")
                return data
            else:
                logger.info("Aucun cache local trouvé, création d'une nouvelle base")
                return {"labels": [], "embeddings": []}
        except Exception as e:
            logger.error(f"Erreur cache local: {e}")
            return {"labels": [], "embeddings": []}
    
    def backup_database(self):
        """Crée une sauvegarde timestampée"""
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_key = f"database/backups/embedding_database_{timestamp}.json"
            
            # Copier la base actuelle vers le backup
            copy_source = {
                'Bucket': self.bucket_name,
                'Key': self.db_key
            }
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=backup_key
            )
            logger.info(f"Backup créé: {backup_key}")
            return backup_key
        except Exception as e:
            logger.error(f"Erreur lors du backup: {e}")
            return None
    
    def get_database_info(self):
        """Informations sur la base de données"""
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=self.db_key
            )
            return {
                "exists": True,
                "last_modified": response['LastModified'],
                "size": response['ContentLength'],
                "location": f"s3://{self.bucket_name}/{self.db_key}"
            }
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                return {
                    "exists": False,
                    "location": f"s3://{self.bucket_name}/{self.db_key}"
                }
            else:
                return {"error": str(e)}

# Instance globale du gestionnaire de base de données
db_manager = S3DatabaseManager()

# Fonctions de compatibilité avec l'ancienne interface
def load_database(db_path=None):
    """Interface de compatibilité pour charger la base de données"""
    return db_manager.load_database()

def save_database(database, db_path=None):
    """Interface de compatibilité pour sauvegarder la base de données"""
    return db_manager.save_database(database)