import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# 1. Setup Paths
MODEL_PATH = 'models/disease_model.h5'
JSON_PATH = 'model_training/class_indices.json'

def predict_image(image_path):
    print(f"🔍 Analyzing image: {image_path}...")
    
    # 2. Load Model and Dictionary
    try:
        model = load_model(MODEL_PATH)
        with open(JSON_PATH, 'r') as f:
            class_indices = json.load(f)
        class_names = {v: k for k, v in class_indices.items()}
    except Exception as e:
        print(f"❌ Error loading model or JSON: {e}")
        return

    # 3. Load and Preprocess the Image
    try:
        img = keras_image.load_img(image_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize to 0-1
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return

    # 4. Make Prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    # 5. Get the Result
    result_name = class_names.get(predicted_class_index, "Unknown")
    formatted_name = result_name.replace("_", " ").title()

    print("\n" + "="*40)
    print("🌾 AI PREDICTION RESULT 🌾")
    print("="*40)
    print(f"Disease:    {formatted_name}")
    print(f"Confidence: {confidence:.2f}%")
    print("="*40 + "\n")

if __name__ == "__main__":
    # ==========================================
    # ⚠️ Change this path to an image on your computer to test it!
    # ==========================================
    TEST_IMAGE_PATH = r"F:\project\smart_crop_advisor\dataset\validation\maize_blight\some_image_name.jpg"
    
    if os.path.exists(TEST_IMAGE_PATH):
        predict_image(TEST_IMAGE_PATH)
    else:
        print(f"❌ Could not find the test image at: {TEST_IMAGE_PATH}")