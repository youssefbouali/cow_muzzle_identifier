# Generated from: model-notebook.ipynb
# Converted at: 2026-02-23T02:05:37.700Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # Import Required Libraries
# This cell imports necessary libraries for model creation, data preprocessing, and evaluation.


from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import json
import numpy as np
from tensorflow.keras.models import load_model
import os
from glob import glob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score

# # Define Base Model
# This cell defines the base VGG16 model and adds custom layers for classification.


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(268, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# # Data Generators
# ##### Dataset donwloading link : https://universe.roboflow.com/iit-jammu-elhdf/cow-face-2qral


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "./train",
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=16
)

train_classes = train_generator.class_indices.keys()

val_generator = val_datagen.flow_from_directory(
    "./valid",
    target_size=(224, 224),
    class_mode='categorical',
    batch_size=16,
    classes=list(train_classes)
)

num_classes = len(train_generator.class_indices)

# # Train Model
# This cell trains the model using the training and validation data.
# 


# model.fit(
#       train_generator,
#       validation_data=val_generator,
#       epochs=50
#   )

# # Save Model
# This cell saves the trained model to a file.


# model.save('muzzle.keras')

# # Load Model and Create Embedding Model
# This cell loads the saved model and creates an embedding model for feature extraction.
# 


model = load_model("muzzle.keras")
embedding_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# ## Function to return embeddings from a muzzle image


def get_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    embedding = embedding_model.predict(img_array)
    return embedding.flatten()

# # Function to build a database that contains the embeddings of each cow 


def build_embedding_database(train_dir="./train"):
    database = {"embeddings": [], "labels": []}
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path): continue
        for img_path in glob(os.path.join(class_path, "*.jpg")):
            emb = get_embedding(img_path)
            database["embeddings"].append(emb)
            database["labels"].append(class_name)
    return database

# ### Building the database and saving it as a json from


def save_database_as_json(database, filename="embedding_database.json"):
    json_ready = {
        "labels": database["labels"],
        "embeddings": [emb.tolist() for emb in database["embeddings"]]
    }
    with open(filename, "w") as f:
        json.dump(json_ready, f)
    print(f"✅ Base sauvegardée dans {filename}")

def load_database_from_json(filename="embedding_database.json"):
    with open(filename, "r") as f:
        data = json.load(f)
    return {
        "labels": data["labels"],
        "embeddings": [np.array(emb) for emb in data["embeddings"]]
    }

# database = build_embedding_database("./train")
# save_database_as_json(database, filename="embedding_database.json")
        # if you saved the dataset already, use this for loading:  
database = load_database_from_json("embedding_database.json")

# # Predict Identity
# This cell defines a function to predict the identity of a cow based on an image and the embedding database.
# 


def predict_identity(img_path, database, seuil=0.7):
    query_emb = get_embedding(img_path)
    sims = cosine_similarity([query_emb], database["embeddings"])[0]
    best_score = np.max(sims)
    best_index = np.argmax(sims)
    
    if best_score < seuil:
        return "INCONNUE", best_score
    else:
        return database["labels"][best_index], best_score

# ## Model testing on test data


def evaluate_on_test_folder(test_dir, database, seuil=0.7):
    y_true = []
    y_pred = []

    for true_label in os.listdir(test_dir):
        class_path = os.path.join(test_dir, true_label)
        if not os.path.isdir(class_path):
            continue

        image_paths = glob(os.path.join(class_path, "*.jpg"))
        for img_path in image_paths:
            predicted_label, score = predict_identity(img_path, database, seuil=seuil)

            y_true.append(true_label)
            y_pred.append(predicted_label)

            print(f"{os.path.basename(img_path)} → vrai: {true_label}, prédit: {predicted_label}, score: {score:.2f}")

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✅ Accuracy sur le dossier de test : {accuracy * 100:.2f}%")

    return accuracy, y_true, y_pred


test_dir = "./test"
accuracy, y_true, y_pred = evaluate_on_test_folder(test_dir, database)

# # Add New Cow to Database
# This cell defines a function to add a new cow's images to the embedding database.


def add_new_cow_to_database(cow_id, image_paths, database):
    for img_path in os.listdir(image_paths):
        emb = get_embedding(os.path.join(image_paths, img_path))
        database["embeddings"].append(emb)
        database["labels"].append(cow_id)

add_new_cow_to_database("cow_test", "./train/cattle_new", database)

# ## To make an individial prediction


img_path = "test/cattle_new/cropped_3.jpg"
label, score = predict_identity(img_path, database)

print(f"🧠 Vache prédite : {label} (confiance : {score:.2f})")

# ## Export to tar


import tarfile
import os
import tensorflow as tf

input_model = '../cow_api/utils/muzzle.keras'   
export_path = './export_temp/1/muzzle.keras'    
output_file = 'model.tar.gz'

if not os.path.exists(export_path):
    print(f"Chargement de {input_model}...")
    model = tf.keras.models.load_model(input_model)
    
    print("Sauvegarde au format SavedModel...")
    model.save(export_path)
else:
    print("Le dossier d'export existe déjà, on passe à la compression.")

print(f"Création de {output_file}...")
with tarfile.open(output_file, "w:gz") as tar:
    tar.add(export_path, arcname='1') 

print("Succès ! Le fichier model.tar.gz est prêt.")