import cv2
import numpy as np
import logging
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO

# Initialize YOLO model
try:
    yolo_model = YOLO("utils/last.pt")
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    yolo_model = None

def preprocess_image(img_np):
    """Prepare image for embedding model (VGG16)"""
    # Resize to 224x224 (expected VGG16 input size)
    img = cv2.resize(img_np, (224, 224))
    # Normalization
    img_array = img.astype(np.float32) / 255.0
    return img_array

def detect_muzzle(image_path, conf=0.5):
    """Detects and crops the cow's muzzle (nose area)"""
    if yolo_model is None:
        logging.error("YOLO model not loaded")
        return None
        
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        logging.warning(f"Could not load image {image_path}")
        return None
        
    results = list(yolo_model(img_cv, conf=conf))
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        h, w, _ = img_cv.shape
        margin = 15
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)
        
        cropped = img_cv[y1:y2, x1:x2]
        return cropped
    return None

def rotate_3d(img, yaw=0, pitch=0, roll=0, f=500):
    """
    Simulates image rotation in 3D space.
    yaw: Horizontal rotation (left/right)
    pitch: Vertical rotation (up/down)
    roll: Image tilt
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    
    # Convert degrees to radians
    yaw_f = np.deg2rad(yaw)
    pitch_f = np.deg2rad(pitch)
    roll_f = np.deg2rad(roll)
    
    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch_f), -np.sin(pitch_f)],
                   [0, np.sin(pitch_f), np.cos(pitch_f)]])
    
    Ry = np.array([[np.cos(yaw_f), 0, np.sin(yaw_f)],
                   [0, 1, 0],
                   [-np.sin(yaw_f), 0, np.cos(yaw_f)]])
    
    Rz = np.array([[np.cos(roll_f), -np.sin(roll_f), 0],
                   [np.sin(roll_f), np.cos(roll_f), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    # Define source points in 2D
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # Define 3D points
    pts = np.array([
        [-cx, -cy, 0],
        [w-cx, -cy, 0],
        [w-cx, h-cy, 0],
        [-cx, h-cy, 0]
    ], dtype=np.float32)
    
    rotated_pts = pts @ R.T
    rotated_pts[:, 2] += f  # Move camera distance
    
    # Project back to 2D
    projected = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        projected[i, 0] = rotated_pts[i, 0] * f / rotated_pts[i, 2] + cx
        projected[i, 1] = rotated_pts[i, 1] * f / rotated_pts[i, 2] + cy
        
    M = cv2.getPerspectiveTransform(src_pts, projected)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def generate_augmentations(muzzle_img):
    """Generates exactly 16 simulated 3D augmented images"""
    augmentations = []
    labels = []
    
    # Image sequence:
    # 1. Original
    augmentations.append(muzzle_img)
    labels.append("original")
    
    # 2. Flipped (Mirror)
    flipped = cv2.flip(muzzle_img, 1)
    augmentations.append(flipped)
    labels.append("flipped")
    
    # 3-6. Yaw variation (Horizontal)
    for yaw in [-25, -12, 12, 25]:
        augmentations.append(rotate_3d(muzzle_img, yaw=yaw))
        labels.append(f"yaw_{yaw}")
        
    # 7-8. Pitch variation (Vertical)
    for pitch in [-18, 18]:
        augmentations.append(rotate_3d(muzzle_img, pitch=pitch))
        labels.append(f"pitch_{pitch}")
        
    # 9-10. Roll variation (Tilt)
    for roll in [-12, 12]:
        augmentations.append(rotate_3d(muzzle_img, roll=roll))
        labels.append(f"roll_{roll}")
        
    # 11-16. Complex 3D movements (Composite Yaw + Pitch)
    composite_angles = [
        (-15, -10), (-15, 10),  # Left-Up, Left-Down
        (15, -10), (15, 10),    # Right-Up, Right-Down
        (0, -20), (0, 20)       # Extreme Vertical
    ]
    
    for yaw, pitch in composite_angles:
        if len(augmentations) >= 16: break
        augmentations.append(rotate_3d(muzzle_img, yaw=yaw, pitch=pitch))
        labels.append(f"3d_y{yaw}_p{pitch}")
        
    # Ensure we return exactly 16 if possible
    return augmentations[:16], labels[:16]