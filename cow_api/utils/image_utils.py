import cv2
import numpy as np
import logging
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO

# Initialiser le modèle YOLO
try:
    yolo_model = YOLO("utils/last.pt")
except Exception as e:
    logging.error(f"Erreur lors du chargement du modèle YOLO: {e}")
    yolo_model = None

def preprocess_image(img_np):
    """Prépare l'image pour le modèle d'embedding (VGG16)"""
    # Redimensionner à 224x224 (taille attendue par VGG16)
    img = cv2.resize(img_np, (224, 224))
    # Normalisation (si nécessaire, selon comment le modèle a été entraîné)
    # Dans le notebook, rescale=1./255 est utilisé dans ImageDataGenerator
    img_array = img.astype(np.float32) / 255.0
    return img_array

def detect_muzzle(image_path, conf=0.5):
    """Détecte et découpe le museau de la vache"""
    if yolo_model is None:
        logging.error("Modèle YOLO non chargé")
        return None
        
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        logging.warning(f"Impossible de charger l'image {image_path}")
        return None
        
    results = list(yolo_model(img_cv, conf=conf))
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        # Ajouter une petite marge si possible
        h, w, _ = img_cv.shape
        margin = 10
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        cropped = img_cv[y1:y2, x1:x2]
        return cropped
    return None

def generate_augmentations(muzzle_img):
    """Génère des versions augmentées du museau (flip, perspective)"""
    augmentations = []
    
    # 1. Image originale
    augmentations.append(muzzle_img)
    
    # 2. Flip horizontal (Effet miroir)
    flipped = cv2.flip(muzzle_img, 1)
    augmentations.append(flipped)
    
    # 3. Perspective Transform - Simulation de regard vers la gauche
    # On étire le côté droit et on compresse le côté gauche
    h, w = muzzle_img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Vers la gauche
    dst_pts_left = np.float32([
        [w*0.1, h*0.05], [w, 0], 
        [w*0.1, h*0.95], [w, h]
    ])
    matrix_left = cv2.getPerspectiveTransform(src_pts, dst_pts_left)
    left_view = cv2.warpPerspective(muzzle_img, matrix_left, (w, h))
    augmentations.append(left_view)
    
    # Vers la droite
    dst_pts_right = np.float32([
        [0, 0], [w*0.9, h*0.05], 
        [0, h], [w*0.9, h*0.95]
    ])
    matrix_right = cv2.getPerspectiveTransform(src_pts, dst_pts_right)
    right_view = cv2.warpPerspective(muzzle_img, matrix_right, (w, h))
    augmentations.append(right_view)
    
    return augmentations