from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import logging

yolo_model = YOLO("utils/last.pt")


def preprocess_image(img_np):
    # img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    img = cv2.resize(img_np, (224, 224))
    # x = image.img_to_array(img) / 255.0
    return img

    
def detect_muzzle(image_path, conf=0.5):
                    # Charger et traiter l'image
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        logging.warning(f"Impossible de charger l'image {image_path}")
        return None
    results = list(yolo_model(img_cv, conf=conf))
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cropped = img_cv[y1:y2, x1:x2]
        return cropped
    return None