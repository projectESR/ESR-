# ==============================================================================
# app.py - Main Flask Application for Deep Learning Blood Grouping
# ==============================================================================

import os
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Configuration & Global Variables ---
MODEL_PATH = 'models/final_blood_grouping_model.h5'
# The model expects images of a specific size. Common sizes are 128x128 or 224x224.
# **IMPORTANT**: Change this if your model was trained on a different image size.
MODEL_IMG_SIZE = (128, 128) 

# --- Load The Deep Learning Model ---
try:
    model = load_model(MODEL_PATH)
    print("✅ Deep Learning model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")
    print("⚠️ Running in fallback mode. All predictions will be 'Undetermined'.")

# --- Blood Type Logic ---
# These are the sections of the test card
ANTIBODY_TYPES = ["Anti A", "Anti B", "Anti D"]
# Rules to determine blood type from test results (True = Agglutination, False = No Agglutination)
BLOOD_TYPE_RULES = {
    # (Anti-A, Anti-B, Anti-D)
    (True, False, True): "A+",
    (True, False, False): "A-",
    (False, True, True): "B+",
    (False, True, False): "B-",
    (True, True, True): "AB+",
    (True, True, False): "AB-",
    (False, False, True): "O+",
    (False, False, False): "O-",
}

# ==============================================================================
# Core Functions
# ==============================================================================

def preprocess_image_section(img_section_pil):
    """
    Prepares an image section for the deep learning model.
    """
    # Resize to the size the model expects
    img_resized = img_section_pil.resize(MODEL_IMG_SIZE)
    # Convert to a NumPy array
    img_array = np.array(img_resized)
    # Ensure it's 3 channels (RGB)
    if img_array.ndim == 2: # if grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    # Normalize pixel values to be between 0 and 1
    img_array = img_array / 255.0
    # Add a "batch" dimension, e.g., (128, 128, 3) -> (1, 128, 128, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

def analyze_single_section(img_section_pil):
    """
    Uses the loaded model to predict agglutination for a single image section.
    """
    if model is None:
        return {"agglutination": None, "confidence": 0}

    # Preprocess the image section
    processed_batch = preprocess_image_section(img_section_pil)
    
    # Get model prediction
    prediction = model.predict(processed_batch)[0][0] # Get the single probability value
    
    # Determine agglutination based on a 0.5 threshold
    agglutination = bool(prediction > 0.5)
    confidence = float(prediction) if agglutination else 1 - float(prediction)

    return {
        "agglutination": agglutination,
        "confidence": round(confidence * 100, 2)
    }

# ==============================================================================
# Flask Routes
# ==============================================================================

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file upload and blood type prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read image in-memory
            image_bytes = file.read()
            img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img = np.array(img_pil) # Convert to numpy array for splitting
            h, w, _ = img.shape

            # --- Split the image into 3 sections ---
            # Assumes the card has 3 circles arranged horizontally
            num_sections = 3
            split_size = w // num_sections
            sections_np = [img[:, i*split_size:(i+1)*split_size] for i in range(num_sections)]
            
            # --- Analyze each section ---
            analysis_results = []
            blood_results_tuple = []
            for i, section_np in enumerate(sections_np):
                section_pil = Image.fromarray(section_np)
                result = analyze_single_section(section_pil)
                
                analysis_results.append({
                    "name": ANTIBODY_TYPES[i],
                    **result # merge the dictionaries
                })
                blood_results_tuple.append(result['agglutination'])
            
            # --- Determine Final Blood Type ---
            final_blood_type = BLOOD_TYPE_RULES.get(tuple(blood_results_tuple), "Undetermined")
            
            # Convert image to base64 to display on the page
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')

            return jsonify({
                'blood_type': final_blood_type,
                'analysis': analysis_results,
                'model_used': 'Deep Learning' if model else 'Model Not Loaded',
                'image_data': img_base64
            })

        except Exception as e:
            print(f"Error during processing: {e}")
            return jsonify({'error': 'An error occurred during image processing.'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    # Set debug=False for production
    app.run(host='0.0.0.0', port=5000, debug=True)