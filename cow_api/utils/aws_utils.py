import boto3
import cv2
import os
from io import BytesIO
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)

class S3Manager:
    def __init__(self, bucket_name=None, region_name='us-east-1'):
        """
        Initialise le gestionnaire S3
        
        Args:
            bucket_name: Nom du bucket S3 (peut être défini via variable d'environnement AWS_S3_BUCKET)
            region_name: Région AWS (défaut: us-east-1)
        """
        self.bucket_name = bucket_name or os.getenv('AWS_S3_BUCKET', 'cow-muzzle-images')
        self.region_name = region_name
        
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
    
    def create_bucket_if_not_exists(self):
        """Crée le bucket S3 s'il n'existe pas"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Le bucket {self.bucket_name} existe déjà")
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                try:
                    if self.region_name == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region_name}
                        )
                    logger.info(f"Bucket {self.bucket_name} créé avec succès")
                except ClientError as create_error:
                    logger.error(f"Erreur lors de la création du bucket: {create_error}")
                    raise
            else:
                logger.error(f"Erreur lors de la vérification du bucket: {e}")
                raise
    
    def list_cow_raw_images(self, cow_id):
        """
        Liste toutes les images brutes d'une vache dans le dossier raw_images
        
        Args:
            cow_id: ID de la vache
            
        Returns:
            list: Liste des clés S3 des images brutes
        """
        try:
            prefix = f"{cow_id}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            image_keys = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    # Filtrer pour ne garder que les fichiers image
                    key = obj['Key']
                    if key.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        image_keys.append(key)
            
            logger.info(f"Trouvé {len(image_keys)} images pour la vache {cow_id}")
            return image_keys
            
        except ClientError as e:
            logger.error(f"Erreur lors de la liste des images brutes: {e}")
            return []
    
    def download_image(self, s3_key, local_path):
        """
        Télécharge une image depuis S3 vers un fichier local
        
        Args:
            s3_key: Clé S3 de l'image
            local_path: Chemin local de destination
            
        Returns:
            bool: True si succès, False sinon
        """
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            logger.info(f"Image téléchargée: {s3_key} -> {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"Erreur lors du téléchargement de {s3_key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Erreur inattendue lors du téléchargement: {e}")
            return False 