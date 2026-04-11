import os
import sqlite3
import hashlib
import secrets
import time
import re
from functools import wraps
from flask import Flask, request, jsonify, session, redirect, url_for, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from datetime import datetime
from dotenv import load_dotenv
import json
from scipy.special import expit as sigmoid  # Sigmoid function for proper normalization
from openai import OpenAI

# MedGemma imports (optional - will gracefully degrade if not available)
try:
    from transformers import pipeline
    import torch
    HAS_MEDGEMMA = True
except ImportError:
    HAS_MEDGEMMA = False
    print("Warning: transformers/torch not installed. MedGemma disabled.")

load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

# MedGemma pipeline (lazy loaded)
medgemma_pipe = None
MEDGEMMA_SYSTEM_PROMPT = """You are an AI-powered ophthalmology screening assistant for EXU Detect.

Your ONLY job is to analyze retinal/eye images for the following conditions:
1. Diabetic Retinopathy (DR) — stages: No DR, Mild, Moderate, Severe, Proliferative
2. Diabetic Macular Edema (DME)
3. Glaucoma indicators (optic disc cupping)
4. Retinal vessel abnormalities
5. Signs of general eye infection or inflammation

You MUST always return your response in this exact JSON format:
{
  "condition_detected": true/false,
  "conditions": [
    {
      "name": "Diabetic Retinopathy",
      "detected": true/false,
      "severity": "None/Mild/Moderate/Severe/Proliferative",
      "confidence": 0-100
    }
  ],
  "overall_risk": "Low/Medium/High/Critical",
  "next_steps": [
    "Step 1 action",
    "Step 2 action"
  ],
  "refer_to_specialist": true/false,
  "retest_in_months": 3/6/12
}

Do NOT analyze anything outside of eye and retinal conditions.
Do NOT ask the patient questions.
Do NOT give general health advice unrelated to eyes."""

def get_medgemma_pipeline():
    global medgemma_pipe
    if medgemma_pipe is None and HAS_MEDGEMMA:
        try:
            print("Loading MedGemma model (this may take a while on first run)...")
            medgemma_pipe = pipeline(
                "image-text-to-text",
                model="google/medgemma-1.5-4b-it",
                torch_dtype=torch.bfloat16,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            print("MedGemma model loaded successfully!")
        except Exception as e:
            print(f"MedGemma load error: {e}")
    return medgemma_pipe

# Database setup
DB_PATH = os.path.join(os.path.dirname(__file__), 'ocuscan.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            module TEXT,
            result TEXT,
            severity TEXT,
            confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            user_message TEXT,
            bot_response TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Model Initialization
try:
    interpreter = tf.lite.Interpreter(model_path="eyedetecter.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    output_map = {}
    for i, detail in enumerate(output_details):
        name = detail['name']
        if '1:0' in name: output_map['cataract'] = detail['index']
        elif '1:1' in name: output_map['conjunctivitis'] = detail['index']
        elif '1:2' in name: output_map['diabetic_retinopathy'] = detail['index']
        elif '1:3' in name: output_map['glaucoma'] = detail['index']
    
    print("Model loaded. Outputs:", output_map)
except Exception as e:
    print(f"Model load error: {e}")
    interpreter = None
    output_map = {}

# NVIDIA Integrate API for Iris chatbot
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY', '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')
NVIDIA_API_URL = 'https://integrate.api.nvidia.com/v1'
NVIDIA_STREAM = True  # Enable streaming by default

IRIS_SYSTEM_PROMPT = """You are Iris, an AI eye health assistant for OcuScan. You help users understand eye diseases, symptoms, and general eye health.

Guidelines:
- Be friendly, empathetic, and professional
- Provide clear, easy-to-understand explanations
- ALWAYS remind users you're for educational purposes, NOT medical diagnosis
- Recommend consulting ophthalmologists for actual medical concerns
- Keep responses concise (2-3 paragraphs max)
- If asked about specific diagnoses, suggest appropriate medical consultation
- Never make definitive medical claims
"""

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function

def login_required_html(f):
    """Decorator for HTML routes - redirects to login if not authenticated"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

# Static files
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

# Auth routes
@app.route('/')
def index():
    if 'user_id' in session:
        return send_from_directory('.', 'index.html')
    return redirect('/login')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/login')
def login_page():
    return send_from_directory('templates/auth', 'login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')
    return send_from_directory('.', 'index.html')

@app.route('/scan/details')
def scan_details():
    if 'user_id' not in session:
        return redirect('/login')
    return send_from_directory('templates/scan', 'details.html')

@app.route('/scan/capture')
def scan_capture():
    if 'user_id' not in session:
        return redirect('/login')
    if 'patient' not in session:
        return redirect('/scan/details')
    return send_from_directory('templates/scan', 'capture.html')

@app.route('/results')
def results_page():
    if 'user_id' not in session:
        return redirect('/login')
    return send_from_directory('templates/results', 'results.html')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    full_name = data.get('full_name', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not full_name or not email or not password:
        return jsonify({"error": "All fields required"}), 400
    
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    
    try:
        conn = get_db()
        c = conn.cursor()
        # Check if email already exists
        c.execute('SELECT id FROM users WHERE email = ?', (email,))
        if c.fetchone():
            conn.close()
            return jsonify({"error": "Email already registered. Try logging in."}), 400
        # Check if username already exists
        c.execute('SELECT id FROM users WHERE username = ?', (full_name,))
        if c.fetchone():
            conn.close()
            return jsonify({"error": "Username already taken. Please choose a different name."}), 400
        # Insert new user
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                  (full_name, email, hash_password(password)))
        conn.commit()
        conn.close()
        return jsonify({"message": "Account created successfully"})
    except sqlite3.IntegrityError as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ? AND password_hash = ?',
              (email, hash_password(password)))
    user = c.fetchone()
    conn.close()
    
    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['email'] = user['email']
        return jsonify({"message": "Login successful", "username": user['username']})
    
    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})

@app.route('/api/me', methods=['GET'])
def me():
    if 'user_id' in session:
        return jsonify({
            "user_id": session['user_id'], 
            "username": session.get('username', ''),
            "email": session.get('email', '')
        })
    return jsonify({"user_id": None})

# Patient details endpoint
@app.route('/api/patient-details', methods=['POST'])
@login_required
def save_patient_details():
    data = request.json
    full_name = data.get('full_name', '').strip()
    age = data.get('age')
    gender = data.get('gender', '').strip()
    blood_group = data.get('blood_group', '').strip()
    phone = data.get('phone', '').strip()
    
    if not full_name or not age or not gender or not blood_group:
        return jsonify({"error": "All required fields must be filled"}), 400
    
    try:
        age = int(age)
        if age < 1 or age > 120:
            return jsonify({"error": "Age must be between 1 and 120"}), 400
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid age value"}), 400
    
    session['patient'] = {
        'full_name': full_name,
        'age': age,
        'gender': gender,
        'blood_group': blood_group,
        'phone': phone
    }
    
    return jsonify({"status": "ok", "message": "Patient details saved"})

@app.route('/api/patient-data')
@login_required
def get_patient_data():
    patient = session.get('patient', {})
    return jsonify({"patient": patient})

def validate_eye_image(img):
    """
    Validates if the uploaded image is a clear eye photo or retinal scan.
    Returns dict with 'is_valid', 'error', and 'image_type':
    - 'retinal': orange/red fundus image (for diabetic retinopathy, glaucoma)
    - 'eye_photo': external eye photo (for cataract, conjunctivitis)
    """
    import numpy as np
    
    # Convert to numpy array
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    # Calculate color channels
    red_avg = np.mean(img_array[:, :, 0])
    green_avg = np.mean(img_array[:, :, 1])
    blue_avg = np.mean(img_array[:, :, 2])
    brightness = np.mean(img_array)
    std_dev = np.std(img_array)
    
    # KEY CHECK 0: Detect RETINAL/FUNDUS images (orange/red background)
    # Retinal images typically have warm orange/red tones
    # More lenient detection: check for red/orange dominance
    is_retinal_fundus = (
        (red_avg > green_avg and red_avg > blue_avg) and  # Red dominant
        (red_avg > 60) and  # Not too dark
        (blue_avg < red_avg * 0.8)  # Blue is lower than red
    )
    
    if is_retinal_fundus:
        # This is a retinal fundus image - accept it
        if std_dev > 10 and 15 < brightness < 250:
            return {'is_valid': True, 'error': None, 'image_type': 'retinal'}
    
    # For EXTERNAL EYE photos (close-up of eye, white sclera):
    # Create center region (where pupil would be)
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 3
    
    y, x = np.ogrid[:height, :width]
    center_mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    center_brightness = np.mean(img_array[center_mask])
    outer_brightness = np.mean(img_array[~center_mask])
    
    # KEY CHECK 1: Eye images MUST have a dark center (pupil)
    # If center is NOT darker than surroundings, it's NOT an eye photo
    if outer_brightness > 0 and center_brightness >= outer_brightness:
        return {
            'is_valid': False,
            'error': 'Image does not show an eye. Please upload a clear close-up photo of an eye.',
            'image_type': None
        }
    
    # Calculate contrast ratio
    contrast_ratio = outer_brightness / center_brightness if center_brightness > 0 else 0
    
    # KEY CHECK 2: Minimum contrast required for eye detection
    # Eye photos need visible pupil (dark center) vs sclera/background
    if contrast_ratio < 1.5:
        return {
            'is_valid': False,
            'error': 'Image does not show a clear eye. Please upload a focused eye photo.',
            'image_type': None
        }
    
    # KEY CHECK 3: Center should be reasonably dark (actual pupil)
    # Pupil is typically very dark (brightness < 100)
    if center_brightness > 100:
        return {
            'is_valid': False,
            'error': 'Image center is too bright to be an eye. Please upload a clear eye photo.',
            'image_type': None
        }
    
    # KEY CHECK 4: Overall brightness should be in eye-photo range
    if brightness < 20 or brightness > 240:
        return {
            'is_valid': False,
            'error': 'Image brightness is unsuitable. Please upload a properly lit eye photo.',
            'image_type': None
        }
    
    # KEY CHECK 5: Reject images that are too uniform (likely not eye photos)
    if std_dev < 20:
        return {
            'is_valid': False,
            'error': 'Image appears too uniform. Please upload a clear eye photo.',
            'image_type': None
        }
    
    # KEY CHECK 6: Aspect ratio should be reasonable for eye photos
    aspect_ratio = width / height
    if aspect_ratio < 0.3 or aspect_ratio > 3.0:
        return {
            'is_valid': False,
            'error': 'Invalid image shape. Please upload a properly cropped eye photo.',
            'image_type': None
        }
    
    # All checks passed - this looks like an external eye photo
    return {'is_valid': True, 'error': None, 'image_type': 'eye_photo'}

# Prediction endpoint
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    start_time = time.time()
    try:
        if 'patient' not in session:
            return jsonify({"error": "Patient details required before scan"}), 400
        
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        user_id = session.get('user_id')
        patient = session.get('patient', {})
        file = request.files['image']
        
        # Save image for MedGemma (need to read twice)
        image_bytes = file.read()
        img_pil = Image.open(__import__('io').BytesIO(image_bytes)).convert('RGB')
        
        # Validate if image looks like an eye/retina scan
        validation_result = validate_eye_image(img_pil)
        
        # Debug: Print validation details
        import numpy as np
        debug_brightness = np.mean(np.array(img_pil))
        debug_std = np.std(np.array(img_pil))
        debug_rgb = [np.mean(np.array(img_pil)[:, :, i]) for i in range(3)]
        print(f"🔍 IMAGE VALIDATION DEBUG:")
        print(f"   Size: {np.array(img_pil).shape}")
        print(f"   Brightness: {debug_brightness:.1f}")
        print(f"   Std Dev: {debug_std:.1f}")
        print(f"   RGB: R={debug_rgb[0]:.1f}, G={debug_rgb[1]:.1f}, B={debug_rgb[2]:.1f}")
        print(f"   Validation Result: {validation_result}")
        
        if not validation_result['is_valid']:
            return jsonify({
                "error": validation_result['error'],
                "warning": True,
                "suggestion": "Please upload a clear retinal or anterior eye scan image. Ensure good lighting and focus on the eye."
            }), 400
        
        # LAYER 1: TensorFlow Lite model
        img = img_pil.resize((224, 224))
        input_data = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        raw_cataract = float(interpreter.get_tensor(output_map['cataract'])[0][0])
        raw_conjunctivitis = float(interpreter.get_tensor(output_map['conjunctivitis'])[0][0])
        raw_retinopathy = float(interpreter.get_tensor(output_map['diabetic_retinopathy'])[0][0])
        raw_glaucoma = float(interpreter.get_tensor(output_map['glaucoma'])[0][0])
        
        # Get image type from validation
        image_type = validation_result.get('image_type', 'eye_photo')
        print(f"🔬 IMAGE TYPE DETECTED: {image_type}")
        
        # Apply calibration FIRST to all scores
        raw_scores = {
            "cataract": raw_cataract,
            "conjunctivitis": raw_conjunctivitis,
            "diabetic_retinopathy": raw_retinopathy,
            "glaucoma": raw_glaucoma
        }
        
        calibrated_scores = {
            "cataract": float(sigmoid((raw_cataract - 0.1) * 10)),
            "conjunctivitis": float(sigmoid((raw_conjunctivitis - 0.1) * 10)),
            "diabetic_retinopathy": float(sigmoid((raw_retinopathy - 0.1) * 25)),
            "glaucoma": float(sigmoid((raw_glaucoma - 0.1) * 35))
        }
        
        # EXPLICIT FILTERING: Only relevant conditions based on image type
        # Eye photos can ONLY detect: cataract, conjunctivitis
        # Retinal images can ONLY detect: diabetic_retinopathy, glaucoma
        
        if image_type == 'retinal':
            # Retinal image - only retinal diseases are valid
            res = {
                "cataract": 0.0,  # NEVER show cataract for retinal images
                "conjunctivitis": 0.0,  # NEVER show conjunctivitis for retinal images
                "diabetic_retinopathy": calibrated_scores.get("diabetic_retinopathy", 0),
                "glaucoma": calibrated_scores.get("glaucoma", 0)
            }
        else:
            # Eye photo - only anterior segment diseases are valid
            res = {
                "cataract": calibrated_scores.get("cataract", 0),
                "conjunctivitis": calibrated_scores.get("conjunctivitis", 0),
                "diabetic_retinopathy": 0.0,  # NEVER show diabetic_retinopathy for eye photos
                "glaucoma": 0.0  # NEVER show glaucoma for eye photos
            }
        
        print(f"📊 RAW MODEL OUTPUTS - Cat: {raw_cataract:.6f}, Conj: {raw_conjunctivitis:.6f}, Retino: {raw_retinopathy:.6f}, Glaucoma: {raw_glaucoma:.6f}")
        print(f"✅ FILTERED SCORES - Cat: {res['cataract']:.4f}, Conj: {res['conjunctivitis']:.4f}, Retino: {res['diabetic_retinopathy']:.4f}, Glaucoma: {res['glaucoma']:.4f}")
        
        # Determine overall result
        max_key = max(res, key=res.get) if any(v > 0 for v in res.values()) else None
        max_val = res[max_key] if max_key and res[max_key] > 0 else 0
        
        # LAYER 2: MedGemma analysis (if available)
        medgemma_results = None
        if HAS_MEDGEMMA:
            try:
                pipe = get_medgemma_pipeline()
                if pipe:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": img_pil},
                                {"type": "text", "text": "Analyze this retinal image for eye conditions and return results in the specified JSON format."}
                            ]
                        }
                    ]
                    
                    output = pipe(text=messages, max_new_tokens=2000)
                    medgemma_text = output[0]["generated_text"][-1]["content"]
                    
                    # Parse JSON from MedGemma response
                    json_match = re.search(r'\{.*\}', medgemma_text, re.DOTALL)
                    if json_match:
                        medgemma_results = json.loads(json_match.group())
                        print(f"🧠 MedGemma analysis: {medgemma_results}")
            except Exception as e:
                print(f"MedGemma error: {e}")
                medgemma_results = None
        
        # Classify severity and recommendation
        if max_val == 0 or max_key is None:
            overall_status = "safe"
            status_text = "No Condition Detected"
            severity = "None"
            recommendation = "No signs of the relevant conditions detected. Continue regular eye check-ups."
        elif max_val < 0.5:
            overall_status = "safe"
            status_text = "Normal"
            severity = "Low"
            recommendation = "Minor signs detected. Monitor and retest in 1-2 weeks if symptoms persist."
        elif max_val < 0.7:
            overall_status = "moderate"
            status_text = "Condition Detected"
            severity = "Moderate"
            recommendation = "Consult an ophthalmologist within 48 hours for proper evaluation."
        else:
            overall_status = "infected"
            status_text = "Condition Detected"
            severity = "High"
            recommendation = "Seek medical attention promptly. Contact an ophthalmologist immediately."
        
        # Build enhanced response
        result_data = {
            "scores": res,
            "medgemma_analysis": medgemma_results,
            "primary_condition": max_key if max_key else "none",
            "confidence": round(max_val * 100, 1),
            "overall_status": overall_status,
            "status_text": status_text,
            "severity": severity,
            "recommendation": recommendation,
            "image_type": image_type,  # 'retinal' or 'eye_photo'
            "patient": patient,
            "timestamp": datetime.now().isoformat(),
            "latency_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        # Save to history if logged in
        if user_id:
            conn = get_db()
            c = conn.cursor()
            c.execute('INSERT INTO scan_history (user_id, module, result, severity, confidence) VALUES (?, ?, ?, ?, ?)',
                      (user_id, max_key, status_text, severity, max_val * 100))
            conn.commit()
            conn.close()
        
        # Store in session for results page
        session['last_scan_result'] = result_data
        
        return jsonify(result_data)
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

# Iris chatbot endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Message required"}), 400
        
        user_id = session.get('user_id')
        
        # Get chat history for context
        history = []
        if user_id:
            conn = get_db()
            c = conn.cursor()
            c.execute('SELECT user_message, bot_response FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 10',
                      (user_id,))
            history = [{"user_message": r['user_message'], "bot_response": r['bot_response']} for r in c.fetchall()]
            conn.close()
        
        # Fallback responses when NVIDIA API key is not configured
        if not NVIDIA_API_KEY:
            fallback_responses = {
                "default": "I'm Iris, your AI eye health assistant. For detailed medical queries, please consult a healthcare professional. In the meantime, how can I help you navigate this app?",
                "symptom": "For any eye symptoms or concerns, I recommend consulting an ophthalmologist. Regular eye check-ups are important for early detection of conditions.",
                "disease": "Eye diseases can range from common conditions like conjunctivitis to more serious ones like diabetic retinopathy. A proper diagnosis requires professional examination.",
            }
            # Simple keyword matching
            msg_lower = message.lower()
            if any(w in msg_lower for w in ['symptom', 'pain', 'hurt', 'red', 'swelling']):
                bot_response = fallback_responses['symptom']
            elif any(w in msg_lower for w in ['disease', 'condition', 'diagnosis', 'dr', 'glaucoma', 'cataract']):
                bot_response = fallback_responses['disease']
            else:
                bot_response = fallback_responses['default']
            
            # Save to history
            if user_id:
                conn = get_db()
                c = conn.cursor()
                c.execute('INSERT INTO chat_history (user_id, user_message, bot_response) VALUES (?, ?, ?)',
                          (user_id, message, bot_response))
                conn.commit()
                conn.close()
            return jsonify({"response": bot_response})
        
        # Build messages
        messages = [{"role": "system", "content": IRIS_SYSTEM_PROMPT}]
        for h in reversed(history):
            messages.append({"role": "user", "content": h['user_message']})
            messages.append({"role": "assistant", "content": h['bot_response']})
        messages.append({"role": "user", "content": message})
        
        # Call NVIDIA Integrate API using OpenAI client
        try:
            client = OpenAI(
                base_url=NVIDIA_API_URL,
                api_key=NVIDIA_API_KEY
            )
            
            completion = client.chat.completions.create(
                model="qwen/qwen2.5-coder-32b-instruct",
                messages=messages,
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024,
                stream=True
            )
            
            # Collect streaming response
            bot_response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    bot_response += chunk.choices[0].delta.content
            
            if not bot_response:
                bot_response = "I apologize, but I couldn't generate a response."
            bot_response = bot_response.strip()
        except Exception as e:
            print(f"NVIDIA API error: {e}")
            # Fallback to simple response
            fallback_responses = {
                "default": "I'm Iris, your AI eye health assistant. For detailed medical queries, please consult a healthcare professional. How can I help you with the app?",
                "symptom": "For eye symptoms or concerns, I recommend consulting an ophthalmologist.",
                "disease": "Eye diseases range from common conditions to serious ones. A proper diagnosis requires professional examination.",
            }
            msg_lower = message.lower()
            if any(w in msg_lower for w in ['symptom', 'pain', 'hurt', 'red', 'swelling']):
                bot_response = fallback_responses['symptom']
            elif any(w in msg_lower for w in ['disease', 'condition', 'diagnosis', 'dr', 'glaucoma', 'cataract']):
                bot_response = fallback_responses['disease']
            else:
                bot_response = fallback_responses['default']

        # Save to history
        if user_id:
            conn = get_db()
            c = conn.cursor()
            c.execute('INSERT INTO chat_history (user_id, user_message, bot_response) VALUES (?, ?, ?)',
                      (user_id, message, bot_response))
            conn.commit()
            conn.close()

        return jsonify({"response": bot_response})
            
    except requests.exceptions.RequestException as e:
        print(f"Chat API error: {e}")
        return jsonify({"response": "I'm having trouble connecting to my AI service. Please try again in a moment."}), 503
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": "I'm experiencing technical difficulties. Please try again later."}), 500

# Get user history
@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT module, result, severity, confidence, created_at FROM scan_history WHERE user_id = ? ORDER BY created_at DESC',
              (session['user_id'],))
    scans = [dict(r) for r in c.fetchall()]
    c.execute('SELECT user_message, bot_response, created_at FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 20',
              (session['user_id'],))
    chats = [dict(r) for r in c.fetchall()]
    conn.close()
    return jsonify({"scans": scans, "chats": chats})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
