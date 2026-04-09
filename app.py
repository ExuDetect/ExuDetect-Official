import os
import sqlite3
import secrets
import json
import re
import base64
from functools import wraps
from flask import Flask, request, jsonify, session, redirect, url_for, render_template, send_from_directory, abort
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
from datetime import datetime
from dotenv import load_dotenv
from scipy.special import expit as sigmoid  # Sigmoid function for proper normalization
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

load_dotenv()

# Optional NVIDIA / OpenAI python client (used when available)
try:
    from openai import OpenAI
    NV_OPENAI_CLIENT_AVAILABLE = True
except Exception:
    NV_OPENAI_CLIENT_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))
CORS(app, supports_credentials=True)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Database setup
DB_PATH = os.path.join(os.path.dirname(__file__), 'ocuscan.db')

def get_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        raise

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

# NVIDIA Vision API for eye disease detection
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY') or os.getenv('NV_API_KEY')
NVIDIA_VISION_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'

# OpenRouter API for Iris chatbot (fallback)
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'

# NV / OpenAI integrate API config
NV_OPENAI_API_KEY = os.getenv('NV_OPENAI_API_KEY') or os.getenv('NV_API_KEY') or os.getenv('OPENAI_API_KEY')
NV_OPENAI_BASE = os.getenv('NV_OPENAI_BASE') or 'https://integrate.api.nvidia.com/v1'
# Optional: an NVIDIA NIM / model endpoint you control that returns JSON scores for the 4 detections
# Example: set NV_DETECTION_URL to the model inference URL (this varies by deployment)
NV_DETECTION_URL = os.getenv('NV_DETECTION_URL')

EYE_DETECTION_SYSTEM_PROMPT = """You are an expert ophthalmologist AI assistant specializing in eye disease diagnosis from retinal and eye images.

TASK: Analyze the provided eye image and detect the following conditions with confidence scores (0.0 to 1.0):

1. **CATARACT** - Clouding of the eye's lens
   - Symptoms: Cloudy/milky appearance in pupil, blurred vision indicators
   - Look for: Opacity in lens area, whitish/grayish discoloration
   
2. **DIABETIC RETINOPATHY** - Damage to blood vessels in retina
   - Symptoms: Dark spots, hemorrhages, exudates (white/yellow deposits)
   - Look for: Blood vessel abnormalities, microaneurysms, cotton wool spots
   
3. **GLAUCOMA** - Optic nerve damage from high eye pressure
   - Symptoms: Enlarged optic cup, pale optic disc, cup-to-disc ratio > 0.6
   - Look for: Optic nerve head changes, cupping, nerve fiber layer defects
   
4. **CONJUNCTIVITIS** - Inflammation of conjunctiva
   - Symptoms: Red/pink eye, bloodshot appearance, swelling
   - Look for: Redness in sclera, vascular injection, discharge

RESPONSE FORMAT: Return ONLY a valid JSON object with this exact structure:
{
  "cataract": 0.0,
  "conjunctivitis": 0.0,
  "diabetic_retinopathy": 0.0,
  "glaucoma": 0.0,
  "analysis": "Brief clinical observation"
}

GUIDELINES:
- Scores should be between 0.0 (definitely absent) and 1.0 (definitely present)
- If a condition is clearly visible, score should be > 0.7
- If uncertain but possible signs exist, score 0.3-0.6
- If clearly absent, score < 0.2
- Base analysis on visible clinical signs only
- Be conservative - don't over-diagnose from unclear images
"""

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
    """Generate secure password hash using werkzeug"""
    return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

def verify_password(password, password_hash):
    """Verify password against hash"""
    return check_password_hash(password_hash, password)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({"error": "Authentication required", "code": "UNAUTHORIZED"}), 401
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_email(email):
    """Basic email validation"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

# Static files - More specific route to avoid conflicts
@app.route('/<path:filename>')
def static_files(filename):
    # Prevent serving sensitive files
    if filename.endswith(('.py', '.db', '.env', '.gitignore')):
        abort(403)
    if os.path.exists(filename) and os.path.isfile(filename):
        return send_from_directory('.', filename)
    abort(404)

# Auth routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/auth/<path:filename>')
def auth_static(filename):
    return send_from_directory('templates/auth', filename)

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required", "code": "INVALID_REQUEST"}), 400
            
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validation
        if not username or not email or not password:
            return jsonify({"error": "All fields required", "code": "MISSING_FIELDS"}), 400
        
        if len(username) < 3 or len(username) > 50:
            return jsonify({"error": "Username must be 3-50 characters", "code": "INVALID_USERNAME"}), 400
        
        if not validate_email(email):
            return jsonify({"error": "Invalid email format", "code": "INVALID_EMAIL"}), 400
        
        if len(password) < 6:
            return jsonify({"error": "Password must be at least 6 characters", "code": "WEAK_PASSWORD"}), 400
        
        if len(password) > 128:
            return jsonify({"error": "Password too long", "code": "INVALID_PASSWORD"}), 400
        
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                      (username, email, hash_password(password)))
            conn.commit()
            return jsonify({"message": "Account created successfully", "code": "SUCCESS"}), 201
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return jsonify({"error": "Username already exists", "code": "USERNAME_EXISTS"}), 400
            elif 'email' in str(e):
                return jsonify({"error": "Email already exists", "code": "EMAIL_EXISTS"}), 400
            else:
                return jsonify({"error": "User already exists", "code": "USER_EXISTS"}), 400
        finally:
            conn.close()
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"error": "Internal server error", "code": "SERVER_ERROR"}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required", "code": "INVALID_REQUEST"}), 400
            
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({"error": "Username and password required", "code": "MISSING_FIELDS"}), 400
        
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE username = ?', (username,))
            user = c.fetchone()
            
            if user and verify_password(password, user['password_hash']):
                session['user_id'] = user['id']
                session['username'] = user['username']
                session.permanent = True  # Enable session timeout
                return jsonify({
                    "message": "Login successful", 
                    "code": "SUCCESS",
                    "username": user['username']
                })
            
            return jsonify({"error": "Invalid credentials", "code": "INVALID_CREDENTIALS"}), 401
        finally:
            conn.close()
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"error": "Internal server error", "code": "SERVER_ERROR"}), 500

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"message": "Logged out"})

@app.route('/api/me', methods=['GET'])
def me():
    if 'user_id' in session:
        return jsonify({"user_id": session['user_id'], "username": session['username']})
    return jsonify({"user_id": None})

# Prediction endpoint
@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded", "code": "NO_IMAGE"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected", "code": "NO_IMAGE_SELECTED"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                "code": "INVALID_FILE_TYPE"
            }), 400
        
        user_id = session.get('user_id')

        # Read image bytes
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({"error": "Empty image file", "code": "EMPTY_FILE"}), 400

        raw_cataract = raw_conjunctivitis = raw_retinopathy = raw_glaucoma = 0.0
        used_nv_vision = False

        # Helper to extract scores flexibly from NV response
        def _extract_scores(obj):
            keys = ['cataract', 'conjunctivitis', 'diabetic_retinopathy', 'glaucoma']
            found = {}
            if not obj:
                return found
            # direct keys
            if isinstance(obj, dict):
                for k in keys:
                    if k in obj:
                        try:
                            found[k] = float(obj[k])
                        except Exception:
                            pass
                # predictions list
                if 'predictions' in obj and isinstance(obj['predictions'], list) and len(obj['predictions'])>0:
                    p = obj['predictions'][0]
                    if isinstance(p, dict):
                        for k in keys:
                            if k in p and k not in found:
                                try:
                                    found[k] = float(p[k])
                                except Exception:
                                    pass
            return found

        # PRIORITY 1: Try NVIDIA Kimi K2.5 Vision API (most accurate for symptom detection)
        if NVIDIA_API_KEY and not used_nv_vision:
            try:
                image_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                headers = {
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "moonshotai/kimi-k2.5",
                    "messages": [
                        {
                            "role": "system",
                            "content": EYE_DETECTION_SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": f"Analyze this eye image for disease symptoms and return JSON scores: <img src=\"data:image/jpeg;base64,{image_b64}\" />"
                        }
                    ],
                    "max_tokens": 1024,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stream": False
                }
                
                print("🔍 Calling NVIDIA Kimi K2.5 Vision API...")
                resp = requests.post(NVIDIA_VISION_URL, headers=headers, json=payload, timeout=45)
                
                if resp.ok:
                    result = resp.json()
                    # Extract content from response
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    print(f"📝 Vision API raw response: {content[:200]}...")
                    
                    # Parse JSON from response (might be in markdown code block)
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        try:
                            scores = json.loads(json_match.group())
                            if all(k in scores for k in ['cataract', 'conjunctivitis', 'diabetic_retinopathy', 'glaucoma']):
                                raw_cataract = scores['cataract']
                                raw_conjunctivitis = scores['conjunctivitis']
                                raw_retinopathy = scores['diabetic_retinopathy']
                                raw_glaucoma = scores['glaucoma']
                                used_nv_vision = True
                                print(f'✅ NVIDIA Vision API success - Cat: {raw_cataract:.3f}, Conj: {raw_conjunctivitis:.3f}, Retino: {raw_retinopathy:.3f}, Glaucoma: {raw_glaucoma:.3f}')
                            else:
                                print(f'⚠️ Vision API returned JSON but missing keys: {scores}')
                                print(f'   Expected keys: cataract, conjunctivitis, diabetic_retinopathy, glaucoma')
                        except json.JSONDecodeError as json_err:
                            print(f'⚠️ Invalid JSON in vision API response: {json_err}')
                            print(f'   Raw content: {content[:300]}')
                    else:
                        print(f'⚠️ Could not parse JSON from vision API response')
                        print(f'   Raw content: {content[:300]}')
                else:
                    print(f'❌ NVIDIA Vision API HTTP {resp.status_code}: {resp.text[:200]}')
            except Exception as e:
                print(f'❌ NVIDIA Vision API request failed: {e}')
                import traceback
                traceback.print_exc()

        # PRIORITY 2: Try NVIDIA-hosted detection endpoint if provided
        if NV_DETECTION_URL and NV_OPENAI_API_KEY and not used_nv_vision:
            try:
                headers = {'Authorization': f'Bearer {NV_OPENAI_API_KEY}'}
                files = {'image': ('image.jpg', img_bytes, 'image/jpeg')}
                resp = requests.post(NV_DETECTION_URL, headers=headers, files=files, timeout=30)
                if resp.ok:
                    j = resp.json()
                    scores = _extract_scores(j)
                    if all(k in scores for k in ['cataract','conjunctivitis','diabetic_retinopathy','glaucoma']):
                        raw_cataract = scores['cataract']
                        raw_conjunctivitis = scores['conjunctivitis']
                        raw_retinopathy = scores['diabetic_retinopathy']
                        raw_glaucoma = scores['glaucoma']
                        used_nv_vision = True
                        print('🔎 Used NV detection endpoint, raw scores:', scores)
                    else:
                        print('NV detection returned JSON but missing keys:', j)
                else:
                    print(f'NV detection HTTP {resp.status_code}: {resp.text}')
            except Exception as e:
                print('NV detection request failed:', e)

        # PRIORITY 3: Fall back to local TFLite interpreter if vision APIs failed
        if not used_nv_vision:
            if interpreter is None:
                return jsonify({
                    "error": "No detection model available. Please configure NVIDIA API key.",
                    "code": "MODEL_UNAVAILABLE"
                }), 503
            
            try:
                print("⚙️ Using local TFLite model as fallback...")
                # Load PIL image from bytes
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                img = img.resize((224, 224))
                input_data = np.array(img, dtype=np.float32) / 255.0
                input_data = np.expand_dims(input_data, axis=0)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                raw_cataract = float(interpreter.get_tensor(output_map['cataract'])[0][0])
                raw_conjunctivitis = float(interpreter.get_tensor(output_map['conjunctivitis'])[0][0])
                raw_retinopathy = float(interpreter.get_tensor(output_map['diabetic_retinopathy'])[0][0])
                raw_glaucoma = float(interpreter.get_tensor(output_map['glaucoma'])[0][0])
            except Exception as model_err:
                print(f"Model inference error: {model_err}")
                return jsonify({
                    "error": "Failed to process image",
                    "code": "INFERENCE_ERROR",
                    "details": str(model_err)
                }), 500
        
        print(f"📊 RAW MODEL OUTPUTS - Cat: {raw_cataract:.6f}, Conj: {raw_conjunctivitis:.6f}, Retino: {raw_retinopathy:.6f}, Glaucoma: {raw_glaucoma:.6f}")
        
        # Adaptive Calibration - Using sigmoid with better parameters
        # The model outputs small values, so we use a lower offset (0.1) and higher gain (10)
        # This amplifies small differences while keeping values in 0-1 range
        res = {
            "cataract": float(sigmoid((raw_cataract - 0.1) * 10)),
            "conjunctivitis": float(sigmoid((raw_conjunctivitis - 0.1) * 10)),
            "diabetic_retinopathy": float(sigmoid((raw_retinopathy - 0.1) * 25)),
            "glaucoma": float(sigmoid((raw_glaucoma - 0.1) * 10))
        }
        
        # Ensure valid range without capping
        for key in res:
            res[key] = max(0.0, min(1.0, res[key]))
        
        print(f"✅ CALIBRATED SCORES - Cat: {res['cataract']:.4f}, Conj: {res['conjunctivitis']:.4f}, Retino: {res['diabetic_retinopathy']:.4f}, Glaucoma: {res['glaucoma']:.4f}")
        
        # Save to history if logged in
        if user_id:
            try:
                max_key = max(res, key=res.get)
                max_val = res[max_key]
                severity = "Severe" if max_val > 0.85 else "Moderate" if max_val > 0.65 else "Mild"
                conn = get_db()
                try:
                    c = conn.cursor()
                    c.execute('INSERT INTO scan_history (user_id, module, result, severity, confidence) VALUES (?, ?, ?, ?, ?)',
                              (user_id, max_key, "Detected", severity, max_val * 100))
                    conn.commit()
                finally:
                    conn.close()
            except Exception as db_err:
                print(f"Failed to save scan history: {db_err}")
                # Don't fail the request if history save fails
        
        return jsonify({
            "code": "SUCCESS",
            "scores": res,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            "error": "Failed to analyze image",
            "code": "PREDICTION_ERROR",
            "details": str(e) if app.debug else None
        }), 500

# Iris chatbot endpoint
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Request body required", "code": "INVALID_REQUEST"}), 400
            
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"error": "Message required", "code": "EMPTY_MESSAGE"}), 400
        
        if len(message) > 1000:
            return jsonify({"error": "Message too long (max 1000 chars)", "code": "MESSAGE_TOO_LONG"}), 400
        
        user_id = session.get('user_id')
        
        # Get chat history for context
        history = []
        if user_id:
            try:
                conn = get_db()
                try:
                    c = conn.cursor()
                    c.execute('SELECT user_message, bot_response FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 10',
                              (user_id,))
                    history = [{"user_message": r['user_message'], "bot_response": r['bot_response']} for r in c.fetchall()]
                finally:
                    conn.close()
            except Exception as db_err:
                print(f"Failed to load chat history: {db_err}")
        
        # Build messages
        messages = [{"role": "system", "content": IRIS_SYSTEM_PROMPT}]
        for h in reversed(history):
            messages.append({"role": "user", "content": h['user_message']})
            messages.append({"role": "assistant", "content": h['bot_response']})
        messages.append({"role": "user", "content": message})
        
        # Prefer NV / OpenAI integrate client when available
        bot_response = None
        if NV_OPENAI_CLIENT_AVAILABLE and NV_OPENAI_API_KEY:
            try:
                client = OpenAI(base_url=NV_OPENAI_BASE, api_key=NV_OPENAI_API_KEY)
                completion = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7,
                    stream=False
                )

                # Attempt to extract text from the returned object (supports both dict-like and attr-like)
                try:
                    # dataclass-like object
                    ch = completion.choices[0]
                    if getattr(ch, 'message', None) and getattr(ch.message, 'content', None):
                        bot_response = ch.message.content
                    elif getattr(ch, 'delta', None) and getattr(ch.delta, 'content', None):
                        bot_response = ch.delta.content
                except Exception:
                    try:
                        # dict-like
                        bot_response = completion['choices'][0]['message']['content']
                    except Exception:
                        bot_response = str(completion)

            except Exception as e:
                print(f"NV OpenAI client error: {e}")

        # Fallback to OpenRouter (requests) if NV client not available or failed
        if not bot_response:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "http://localhost:5000",
                "X-Title": "OcuScan Iris"
            }

            payload = {
                "model": "deepseek/deepseek-chat-v3-0324",
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.7
            }

            try:
                response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
                if response.status_code == 200:
                    result = response.json()
                    bot_response = result['choices'][0]['message']['content']
                else:
                    print(f"OpenRouter API Error: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"OpenRouter request failed: {e}")

        if bot_response:
            # Save to history
            if user_id:
                try:
                    conn = get_db()
                    try:
                        c = conn.cursor()
                        c.execute('INSERT INTO chat_history (user_id, user_message, bot_response) VALUES (?, ?, ?)',
                                  (user_id, message, bot_response))
                        conn.commit()
                    finally:
                        conn.close()
                except Exception as db_err:
                    print(f"Failed to save chat history: {db_err}")

            return jsonify({
                "response": bot_response,
                "code": "SUCCESS"
            })
        else:
            return jsonify({
                "response": "I'm having trouble responding right now. Please try again.",
                "code": "AI_UNAVAILABLE"
            })
            
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            "response": "I'm experiencing technical difficulties. Please try again later.",
            "code": "CHAT_ERROR"
        }), 500

# Get user history
@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    try:
        conn = get_db()
        try:
            c = conn.cursor()
            c.execute('SELECT module, result, severity, confidence, created_at FROM scan_history WHERE user_id = ? ORDER BY created_at DESC',
                      (session['user_id'],))
            scans = [dict(r) for r in c.fetchall()]
            c.execute('SELECT user_message, bot_response, created_at FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 20',
                      (session['user_id'],))
            chats = [dict(r) for r in c.fetchall()]
            return jsonify({
                "scans": scans,
                "chats": chats,
                "code": "SUCCESS"
            })
        finally:
            conn.close()
    except Exception as e:
        print(f"History fetch error: {e}")
        return jsonify({
            "error": "Failed to fetch history",
            "code": "HISTORY_ERROR"
        }), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
