# Inference Notebook for Chest Pathologies Diagnosis using DenseNet

# 1. Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from keras.applications.densenet import preprocess_input

# 2. Load trained model (update path to your saved .h5 file)
MODEL_PATH = "chest_densenet_model.h5"  # <-- replace with actual saved model path
model = load_model(MODEL_PATH)
print("Model loaded successfully.")

# 3. Preprocessing function
def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# 4. Prediction function
def predict_image(img_path, class_labels=None):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    if class_labels is not None:
        for label, prob in zip(class_labels, preds[0]):
            print(f"{label}: {prob:.4f}")
    else:
        print("Predictions:", preds)
    return preds

# 5. Example usage
# Provide the path to an image for testing
# class_labels = ["label1", "label2", ..., "labelN"]  # <-- replace with actual classes from training
# preds = predict_image("sample_chest_xray.jpg", class_labels)
