# ==============================================================================
# app.py - Main Flask Application
#
# To run this application:
# 1. Make sure you have all requirements installed:
#    pip install -r requirements.txt
# 2. Make sure your trained model 'blood_type_classifier.pkl' is in the same directory.
#    If not, the app will use the rule-based fallback.
# 3. Create a folder named 'templates' in the same directory as this script.
# 4. Inside 'templates', create a file named 'index.html' and paste the HTML code into it.
# 5. Create a folder named 'uploads' for temporary image storage.
# 6. Run this script from your terminal: python app.py
# 7. Open your web browser and go to http://127.0.0.1:5000
# ==============================================================================

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
import base64
import joblib
from skimage.feature import graycomatrix, graycoprops

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Configuration & Global Variables ---
DEBUG = True
MODEL_PATH = 'blood_type_classifier.pkl'
ANTIBODY_TYPES = ["Anti A", "Anti B", "Anti D", "H Antigen Serum Test"]
BLOOD_TYPE_RULES = {
    (False, False, True, True): "Bombay Rh+",
    (False, False, False, True): "Bombay Rh-",
    (False, False, True, False): "O+",
    (False, False, False, False): "O-",
    (True, False, True, False): "A+",
    (True, False, False, False): "A-",
    (False, True, True, False): "B+",
    (False, True, False, False): "B-",
    (True, True, True, False): "AB+",
    (True, True, False, False): "AB-"
}

# --- Load the Machine Learning Model ---
try:
    model = joblib.load(MODEL_PATH)
    if DEBUG:
        print(f"Successfully loaded pre-trained model from '{MODEL_PATH}'")
except FileNotFoundError:
    model = None
    if DEBUG:
        print(f"Warning: No pre-trained model found at '{MODEL_PATH}'. Using rule-based detection.")

# ==============================================================================
# CORE IMAGE PROCESSING AND ML LOGIC (Adapted from your code)
# ==============================================================================

def process_image_for_features(img):
    """Preprocesses an image section for feature extraction."""
    # Convert to HSV and get the saturation channel, which is great for contrast
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # Use Otsu's method to find an optimal global threshold
    _, thresh = cv2.threshold(saturation, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    
    # Apply adaptive thresholding to handle lighting variations
    adaptive = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
    
    # Use morphological closing to fill small holes and consolidate clumps
    kernel = np.ones((5, 5), np.uint8)
    processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    
    return processed

def extract_features(processed_img):
    """Extracts texture features from the processed image for the ML model."""
    # Ensure image is 8-bit integer, as required by graycomatrix
    processed_img = processed_img.astype(np.uint8)

    # Calculate GLCM
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(processed_img, distances=[1], angles=angles, symmetric=True, normed=True)
    
    # Calculate texture properties
    contrast = graycoprops(glcm, 'contrast').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    
    # Add another simple feature: percentage of white pixels (clumping)
    white_percentage = np.sum(processed_img == 255) / processed_img.size
    
    features = [contrast, energy, homogeneity, correlation, dissimilarity, white_percentage]
    if DEBUG:
        print(f"Extracted features for prediction: {features}")
    return np.array(features)

def analyze_single_section(img_section):
    """Analyzes a single image section and returns agglutination status and processed images."""
    # --- Preprocessing ---
    # Create a grayscale version for some operations
    gray = cv2.cvtColor(img_section, cv2.COLOR_BGR2GRAY)
    
    # Get the processed (segmented) image for feature extraction
    segmented = process_image_for_features(img_section)

    # --- Analysis ---
    if model is not None:
        # ML-based prediction
        features = extract_features(segmented)
        # The model expects a 2D array, so we reshape
        prediction = model.predict([features])[0]
        # Assuming the model outputs a simple 0 (no) or 1 (yes) for agglutination.
        agglutination = bool(prediction == 1) 
        if DEBUG:
            print(f"ML predicted agglutination: {agglutination}")
    else:
        # Rule-based fallback (from your original code, slightly adapted)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_std = np.std(hist)
        
        glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        
        # This thresholding logic can be fine-tuned
        agglutination = hist_std < 580 and contrast > 200 and energy < 0.25
        if DEBUG:
            print(f"Rule-based agglutination: {agglutination} (std={hist_std:.2f}, contrast={contrast:.2f}, energy={energy:.2f})")

    # --- Prepare images for frontend ---
    # Encode original section
    _, buffer_orig = cv2.imencode('.png', img_section)
    img_orig_b64 = base64.b64encode(buffer_orig).decode('utf-8')

    # Encode segmented image
    _, buffer_seg = cv2.imencode('.png', segmented)
    img_seg_b64 = base64.b64encode(buffer_seg).decode('utf-8')

    return {
        "agglutination": agglutination,
        "original_b64": img_orig_b64,
        "segmented_b64": img_seg_b64
    }

# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route('/')
def index():
    """Renders the main web page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles the image upload and analysis API endpoint."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the image file in memory
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        # --- Split the image into 4 sections ---
        h, w = img.shape[:2]
        # Determine stacking orientation based on aspect ratio
        aspect_ratio = w / h
        if aspect_ratio > 2:  # Horizontal stacking
            num_sections = 4
            split_size = w // num_sections
            sections = [img[:, i*split_size:(i+1)*split_size] for i in range(num_sections)]
        else:  # Vertical stacking
            num_sections = 4
            split_size = h // num_sections
            sections = [img[i*split_size:(i+1)*split_size, :] for i in range(num_sections)]

        if len(sections) != 4:
            return jsonify({'error': 'Image could not be split into 4 sections'}), 400

        # --- Analyze each section ---
        analysis_results = []
        blood_results_tuple = []
        for i, section in enumerate(sections):
            result = analyze_single_section(section)
            analysis_results.append({
                "name": ANTIBODY_TYPES[i],
                **result # merge the dictionaries
            })
            blood_results_tuple.append(result['agglutination'])
        
        # --- Determine Final Blood Type ---
        final_blood_type = BLOOD_TYPE_RULES.get(tuple(blood_results_tuple), "Undetermined")
        
        return jsonify({
            'blood_type': final_blood_type,
            'analysis': analysis_results,
            'model_used': 'Machine Learning' if model else 'Rule-Based Fallback'
        })

    return jsonify({'error': 'An unknown error occurred'}), 500


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    # Set debug=False for production
    app.run(debug=True, host='0.0.0.0')

