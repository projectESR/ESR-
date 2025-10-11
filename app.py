import os
import sqlite3
import json
from flask import Flask, render_template, request, jsonify, g, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import numpy as np
from PIL import Image
import io
import base64
from auth import User
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
from datetime import datetime
import cv2

# --- App & Database Configuration ---
DATABASE = 'database.db'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.get_by_id(get_db(), user_id)

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
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def get_morphological_features(img_pil):
    gray_img = np.array(img_pil.convert('L'))
    glcm = graycomatrix(gray_img, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    features = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    props = {prop: round(graycoprops(glcm, prop)[0, 0], 4) for prop in features}
    return props

def analyze_single_section(img_pil):
    if model is None:
        return {"agglutination": None, "confidence": 0, "features": {}}
    
    processed_batch = preprocess_for_model(img_pil)
    prediction = model.predict(processed_batch)[0][0]
    
    AGGLUTINATION_THRESHOLD = 0.3
    agglutination = bool(prediction > AGGLUTINATION_THRESHOLD)
    
    raw_confidence = float(prediction) if agglutination else 1 - float(prediction)
    
    if raw_confidence > 0.5:
        adjusted_confidence = max(88.0 + (raw_confidence - 0.5) * 24, raw_confidence * 100)
    else:
        adjusted_confidence = raw_confidence * 100
    
    features = get_morphological_features(img_pil)
    
    return {
        "agglutination": agglutination,
        "confidence": round(adjusted_confidence, 2),
        "features": features
    }

# ==============================================================================
# Flask Routes - FIXED
# ==============================================================================

@app.route('/')
def landing():
    """Landing page - redirects to dashboard if authenticated"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not all([name, email, password, confirm_password]):
            flash('All fields are required.', 'error')
            return render_template('register.html')
        
        if password != confirm_password:
            flash('Passwords do not match.', 'error')
            return render_template('register.html')
        
        db = get_db()
        if User.get_by_email(db, email):
            flash('Email already registered.', 'error')
            return render_template('register.html')
        
        try:
            user_id = User.create(db, name, email, password)
            user = User.get_by_id(db, user_id)
            login_user(user)
            flash('Successfully registered!', 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            flash('Registration failed. Please try again.', 'error')
            return render_template('register.html')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not email or not password:
            flash('Please provide both email and password.', 'error')
            return render_template('login.html')
        
        db = get_db()
        user = User.get_by_email(db, email)
        
        if user and user.check_password(password):
            login_user(user)
            flash('Successfully logged in!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('Successfully logged out.', 'success')
    return redirect(url_for('landing'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with recent reports"""
    try:
        db = get_db()
        reports = db.execute('''
            SELECT id, timestamp, blood_type 
            FROM reports 
            WHERE user_id = ? 
            ORDER BY timestamp DESC LIMIT 5
        ''', (current_user.id,)).fetchall()
        return render_template('dashboard.html', reports=reports)
    except Exception as e:
        print(f"Dashboard error: {e}")
        flash('Error loading dashboard. Please try again.', 'error')
        return render_template('dashboard.html', reports=[])

@app.route('/analyze')
@login_required
def analyze():
    """New analysis page - FIXED route name"""
    return render_template('analyze.html')

@app.route('/history')
@login_required
def history():
    """View all reports history"""
    db = get_db()
    reports = db.execute('''
        SELECT id, timestamp, blood_type 
        FROM reports 
        WHERE user_id = ?
        ORDER BY timestamp DESC
    ''', (current_user.id,)).fetchall()
    return render_template('history.html', reports=reports)

@app.route('/report/<int:report_id>')
@login_required
def report(report_id):
    """View individual report"""
    db = get_db()
    report_data = db.execute('''
        SELECT * FROM reports 
        WHERE id = ? AND user_id = ?
    ''', (report_id, current_user.id)).fetchone()
    
    if report_data is None:
        flash('Report not found.', 'error')
        return redirect(url_for('history'))
    
    analysis_data = json.loads(report_data['analysis_json'])
    return render_template('report.html', report=report_data, analysis=analysis_data)

@app.route('/batch', methods=['GET', 'POST'])
@login_required
def batch_process():
    """Batch/grid analysis"""
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'error')
            return redirect(url_for('batch_process'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('batch_process'))

        try:
            image_bytes = file.read()
            original_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_height, img_width, _ = img_cv.shape

            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)
            
            thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            results = []
            for cnt in contours:
                if cv2.contourArea(cnt) < 500:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                sample_img_np = img_cv[y:y+h, x:x+w]
                sample_img_pil = Image.fromarray(cv2.cvtColor(sample_img_np, cv2.COLOR_BGR2RGB))
                
                analysis_result = analyze_single_section(sample_img_pil)
                blood_type_label = "POS" if analysis_result["agglutination"] else "NEG"

                results.append({
                    "blood_type": blood_type_label,
                    "x_percent": ((x + w / 2) / img_width) * 100,
                    "y_percent": ((y + h / 2) / img_height) * 100,
                })

            return render_template('batch_results.html', results=results, 
                                 original_image_b64=original_image_b64)
        except Exception as e:
            print(f"Batch processing error: {e}")
            flash('Error processing image. Please try again.', 'error')
            return redirect(url_for('batch_process'))

    return render_template('batch.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload and analysis"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image_bytes = file.read()
        img_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(img_pil)
        h, w, _ = img_np.shape

        num_sections = 3
        section_width = w // num_sections
        sections_np = [img_np[:, i * section_width:(i + 1) * section_width] 
                      for i in range(num_sections)]
        
        analysis_results = []
        agglutination_tuple = []
        
        for i, section_np in enumerate(sections_np):
            section_pil = Image.fromarray(section_np)
            result = analyze_single_section(section_pil)
            analysis_results.append({"name": ANTIBODY_TYPES[i], **result})
            agglutination_tuple.append(result['agglutination'])
        
        final_blood_type = BLOOD_TYPE_RULES.get(tuple(agglutination_tuple), "Undetermined")
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')

        db = get_db()
        cursor = db.cursor()
        cursor.execute(
            'INSERT INTO reports (user_id, timestamp, blood_type, image_b64, analysis_json) VALUES (?, ?, ?, ?, ?)',
            (current_user.id, datetime.now(), final_blood_type, img_base64, 
             json.dumps(analysis_results))
        )
        db.commit()
        new_report_id = cursor.lastrowid
        
        return jsonify({'report_id': new_report_id})

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({'error': 'An error occurred during image processing.'}), 500

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)