from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import pandas as pd
import os
import requests
import json
from dotenv import load_dotenv

# --- AI IMPORTS ---
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import io

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'super_secret_fallback_key')

# --- CONFIGURATION ---
# Check your VS Code sidebar: rename 'user.xlsx' to 'users.xlsx' if needed
EXCEL_FILE = 'users.xlsx'

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
MARKET_API_KEY = os.getenv('MARKET_API_KEY')

# ==========================================
# 1. AI MODEL & TREATMENT DICTIONARY
# ==========================================
TREATMENT_DICT = {
    "cotton_disease": "Apply copper-based fungicides. Ensure proper field drainage.",
    "cotton_healthy": "Crop is healthy! Continue standard care.",
    "tomato_bacterial_spot": "Use copper sprays. Avoid overhead watering.",
    "tomato_healthy": "Crop is healthy! Maintain nutrients.",
    "wheat_rust": "Apply triazole fungicides immediately.",
    "wheat_healthy": "Crop is healthy!",
    "rice_leaf_blast": "Apply tricyclazole. Avoid excessive nitrogen.",
    "rice_healthy": "Crop is healthy!",
    "maize_blight": "Apply foliar fungicides. Consider crop rotation.",
    "maize_healthy": "Crop is healthy!"
}

try:
    disease_model = load_model('models/disease_model.h5')
    with open('model_training/class_indices.json', 'r') as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
    print("✅ AI Model Loaded.")
except Exception as e:
    print(f"⚠️ AI Model Offline: {e}")
    disease_model = None

# ==========================================
# 2. DATABASE & AUTHENTICATION
# ==========================================
def init_db():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=['Name', 'Password'])
        df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')

@app.route('/', methods=['GET', 'POST'])
def login():
    init_db()
    if request.method == 'POST':
        action = request.form.get('action')
        name = request.form.get('name')
        password = request.form.get('password')
        try:
            df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
        except:
            df = pd.DataFrame(columns=['Name', 'Password'])

        if action == 'signup':
            if name in df['Name'].values:
                flash("User already exists!")
            else:
                new_user = pd.DataFrame({'Name': [name], 'Password': [password]})
                df = pd.concat([df, new_user], ignore_index=True)
                df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
                session['user'] = name
                return redirect(url_for('dashboard'))
        elif action == 'login':
            user = df[(df['Name'] == name) & (df['Password'] == password)]
            if not user.empty:
                session['user'] = name
                return redirect(url_for('dashboard'))
            else:
                flash("Invalid Credentials.")
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session: return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# ==========================================
# 3. CORE FEATURES
# ==========================================

@app.route('/suggest_crop', methods=['POST'])
def suggest_crop():
    # Fix the 4.9 degree error by defaulting to your local area if GPS is missing
    lat = request.form.get('lat')
    lon = request.form.get('lon')
    if not lat or lat == "0" or lat == "undefined":
        lat, lon = "19.8762", "75.3433" 
    
    soil_type = request.form.get('soil_type', 'loamy').strip().lower()
    
    SOIL_LOGIC = {
        "sandy": "Watermelon, Groundnut, or Gram",
        "loamy": "Wheat, Cotton, or Sugarcane",
        "clay": "Paddy (Rice) or Jute",
        "black": "Cotton or Soybeans",
        "red": "Groundnut, Ragi, or Tobacco"
    }
    recommendation = SOIL_LOGIC.get(soil_type, "Wheat (General Suggestion)")

    weather_url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    try:
        res = requests.get(weather_url).json()
        temp = res.get('main', {}).get('temp', 0)
        hum = res.get('main', {}).get('humidity', 0)
        return jsonify({
            "status": "success", 
            "temperature": f"{round(temp, 1)}", 
            "humidity": hum,
            "soil_type": soil_type.title(),
            "suggested_crop": f"{recommendation} (Based on {soil_type.title()} soil)" 
        })
    except:
        return jsonify({"status": "error", "message": "Weather API Error."})

@app.route('/market_price', methods=['POST'])
def market_price():
    user_input = request.form.get('crop_name', 'Potato').strip()
    clean_name = user_input.title()
    
    # Updated key from .env
    api_key = os.getenv('MARKET_API_KEY')
    
    # Official Resource ID for Daily Mandi Prices
    resource_id = "9ef84268-d588-465a-a308-a864a43d0070"
    
    mandi_url = f"https://api.data.gov.in/resource/{resource_id}?api-key={api_key}&format=json&filters[commodity]={clean_name}"
    
    try:
        response = requests.get(mandi_url, timeout=10)
        data = response.json()
        
        if data.get('records'):
            return jsonify({"status": "success", "data": data['records'][:5], "crop": clean_name})
        else:
            return jsonify({"status": "error", "message": f"No data found for {clean_name}. Try 'Potato'."})
    except Exception as e:
        return jsonify({"status": "error", "message": "API Connection Failed."})

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if not disease_model: return jsonify({"status": "error", "message": "AI Model Offline."})
    file = request.files.get('image')
    if not file: return jsonify({"status": "error", "message": "No image."})
    try:
        img = Image.open(file.stream).convert('RGB').resize((224, 224))
        img_arr = keras_image.img_to_array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        preds = disease_model.predict(img_arr)
        idx = np.argmax(preds)
        name = class_names.get(idx, "Unknown")
        return jsonify({
            "status": "success", 
            "disease": name.replace("_", " ").title(),
            "advice": TREATMENT_DICT.get(name, "Consult an expert.")
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)