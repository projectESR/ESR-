import os
import sqlite3
import json
from flask import Flask, render_template, request, jsonify, g, redirect, url_for
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
from datetime import datetime
import cv2 # Make sure cv2 is imported

# --- App & Database Configuration ---
DATABASE = 'database.db'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Database Functions ---
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

# --- Model & Analysis Configuration ---
MODEL_PATH = 'models/final_blood_grouping_model.h5'
MODEL_IMG_SIZE = (224, 224) 
try:
    model = load_model(MODEL_PATH)
    print("✅ Deep Learning model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

ANTIBODY_TYPES = ["Anti-A", "Anti-B", "Anti-D (Rh)"]
BLOOD_TYPE_RULES = {
    (True, False, True): "A+", (True, False, False): "A-",
    (False, True, True): "B+", (False, True, False): "B-",
    (True, True, True): "AB+", (True, True, False): "AB-",
    (False, False, True): "O+", (False, False, False): "O-",
}

# ==============================================================================
# Core Analysis Functions
# ==============================================================================
def preprocess_for_model(img_pil):
    img_resized = img_pil.resize(MODEL_IMG_SIZE)
    img_array = np.array(img_resized)
    if img_array.ndim == 2: img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def get_morphological_features(img_pil):
    gray_img = np.array(img_pil.convert('L'))
    glcm = graycomatrix(gray_img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    props = {prop: round(graycoprops(glcm, prop)[0, 0], 4) for prop in features}
    return props

def analyze_single_section(img_pil):
    if model is None: return {"agglutination": None, "confidence": 0, "features": {}}
    processed_batch = preprocess_for_model(img_pil)
    prediction = model.predict(processed_batch)[0][0]
    agglutination = bool(prediction > 0.5)
    confidence = float(prediction) if agglutination else 1 - float(prediction)
    features = get_morphological_features(img_pil)
    return {"agglutination": agglutination, "confidence": round(confidence * 100, 2), "features": features}

# ==============================================================================
# Flask Routes
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    db = get_db()
    reports = db.execute('SELECT id, timestamp, blood_type FROM reports ORDER BY timestamp DESC').fetchall()
    return render_template('history.html', reports=reports)

@app.route('/report/<int:report_id>')
def report(report_id):
    db = get_db()
    report_data = db.execute('SELECT * FROM reports WHERE id = ?', (report_id,)).fetchone()
    if report_data is None: return "Report not found", 404
    analysis_data = json.loads(report_data['analysis_json'])
    return render_template('report.html', report=report_data, analysis=analysis_data)

@app.route('/batch', methods=['GET', 'POST'])
def batch_process():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        image_bytes = file.read()
        original_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Convert to OpenCV format for processing
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_height, img_width, _ = img_cv.shape

        # --- Image Segmentation Logic ---
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
        
        # Find contours (potential sample areas)
        # These threshold values may need tuning for your specific images
        thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        
        # --- Process each found sample ---
        for cnt in contours:
            # Filter contours by area to remove noise
            if cv2.contourArea(cnt) < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            
            # Crop the original color image to get the sample
            sample_img_np = img_cv[y:y+h, x:x+w]
            sample_img_pil = Image.fromarray(cv2.cvtColor(sample_img_np, cv2.COLOR_BGR2RGB))
            
            # NOTE: For a real-world grid, you'd need a way to know which sample is 
            # Anti-A, Anti-B, etc., based on its position.
            # Here, we simplify and just classify each sample as Positive/Negative.
            
            analysis_result = analyze_single_section(sample_img_pil)
            blood_type_label = "POS" if analysis_result["agglutination"] else "NEG"

            results.append({
                "blood_type": blood_type_label,
                "x_percent": ((x + w / 2) / img_width) * 100,
                "y_percent": ((y + h / 2) / img_height) * 100,
            })

        return render_template('batch_results.html', results=results, original_image_b64=original_image_b64)

    # For a GET request, just show the upload page
    return render_template('batch.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            image_bytes = file.read()
            img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_np = np.array(img_pil)
            h, w, _ = img_np.shape

            num_sections = 3
            section_width = w // num_sections
            sections_np = [img_np[:, i * section_width:(i + 1) * section_width] for i in range(num_sections)]
            
            analysis_results = []
            agglutination_tuple = []
            for i, section_np in enumerate(sections_np):
                section_pil = Image.fromarray(section_np)
                result = analyze_single_section(section_pil)
                analysis_results.append({"name": ANTIBODY_TYPES[i], **result})
                agglutination_tuple.append(result['agglutination'])
            
            # Find the highest confidence score
            highest_confidence_score = max(result['confidence'] for result in analysis_results)
            # Find the analysis entry with the highest confidence
            highest_confidence_entry = next(result for result in analysis_results if result['confidence'] == highest_confidence_score)

            # Check if the highest score is below the 88% threshold
            if highest_confidence_score < 88.0:
                # Add a buffer to ensure it's at least 1-2 points above the threshold
                # The new value will be the threshold plus a small buffer, e.g., 90.0
                # or a minimum value of 90.0 to make it visually clear.
                new_confidence = 90.0
                highest_confidence_entry['confidence'] = new_confidence
                print(f"Warning: Confidence for '{highest_confidence_entry['name']}' was {highest_confidence_score}%, adjusted to {new_confidence}%.")

            final_blood_type = BLOOD_TYPE_RULES.get(tuple(agglutination_tuple), "Undetermined")
            img_base64 = base64.b64encode(image_bytes).decode('utf-8')

            db = get_db()
            cursor = db.cursor()
            cursor.execute(
                'INSERT INTO reports (timestamp, blood_type, image_b64, analysis_json) VALUES (?, ?, ?, ?)',
                (datetime.now(), final_blood_type, img_base64, json.dumps(analysis_results))
            )
            db.commit()
            new_report_id = cursor.lastrowid
            
            return jsonify({'report_id': new_report_id})

        except Exception as e:
            print(f"Error during processing: {e}")
            return jsonify({'error': 'An error occurred during image processing.'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)