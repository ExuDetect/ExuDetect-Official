import os
import sqlite3
import hashlib
import secrets
from functools import wraps
from flask import Flask, request, jsonify, session, redirect, url_for, render_template, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from datetime import datetime
from dotenv import load_dotenv
from scipy.special import expit as sigmoid  # Sigmoid function for proper normalization

load_dotenv()

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
CORS(app)

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

# OpenRouter API for Iris chatbot
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = 'https://openrouter.ai/api/v1/chat/completions'

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

# Static files
@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('.', filename)

# Auth routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/auth/<path:filename>')
def auth_static(filename):
    return send_from_directory('templates/auth', filename)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not username or not email or not password:
        return jsonify({"error": "All fields required"}), 400
    
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                  (username, email, hash_password(password)))
        conn.commit()
        conn.close()
        return jsonify({"message": "Account created successfully"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email already exists"}), 400

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password_hash = ?',
              (username, hash_password(password)))
    user = c.fetchone()
    conn.close()
    
    if user:
        session['user_id'] = user['id']
        session['username'] = user['username']
        return jsonify({"message": "Login successful", "username": user['username']})
    
    return jsonify({"error": "Invalid credentials"}), 401

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
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        user_id = session.get('user_id')
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        input_data = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        raw_cataract = float(interpreter.get_tensor(output_map['cataract'])[0][0])
        raw_conjunctivitis = float(interpreter.get_tensor(output_map['conjunctivitis'])[0][0])
        raw_retinopathy = float(interpreter.get_tensor(output_map['diabetic_retinopathy'])[0][0])
        raw_glaucoma = float(interpreter.get_tensor(output_map['glaucoma'])[0][0])
        
        print(f"📊 RAW MODEL OUTPUTS - Cat: {raw_cataract:.6f}, Conj: {raw_conjunctivitis:.6f}, Retino: {raw_retinopathy:.6f}, Glaucoma: {raw_glaucoma:.6f}")
        
        # Adaptive Calibration - Using sigmoid with better parameters
        # The model outputs small values, so we use a lower offset (0.1) and higher gain (10)
        # This amplifies small differences while keeping values in 0-1 range
        res = {
            "cataract": float(sigmoid((raw_cataract - 0.1) * 10)),
            "conjunctivitis": float(sigmoid((raw_conjunctivitis - 0.1) * 10)),
            "diabetic_retinopathy": float(sigmoid((raw_retinopathy - 0.1) * 25)),
            "glaucoma": float(sigmoid((raw_glaucoma - 0.1) * 35))
        }
        
        # Ensure valid range without capping
        for key in res:
            res[key] = max(0.0, min(1.0, res[key]))
        
        print(f"✅ CALIBRATED SCORES - Cat: {res['cataract']:.4f}, Conj: {res['conjunctivitis']:.4f}, Retino: {res['diabetic_retinopathy']:.4f}, Glaucoma: {res['glaucoma']:.4f}")
        
        # Save to history if logged in
        if user_id:
            max_key = max(res, key=res.get)
            max_val = res[max_key]
            severity = "Severe" if max_val > 0.85 else "Moderate" if max_val > 0.65 else "Mild"
            conn = get_db()
            c = conn.cursor()
            c.execute('INSERT INTO scan_history (user_id, module, result, severity, confidence) VALUES (?, ?, ?, ?, ?)',
                      (user_id, max_key, "Detected", severity, max_val * 100))
            conn.commit()
            conn.close()
        
        return jsonify(res)
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
        
        # Build messages
        messages = [{"role": "system", "content": IRIS_SYSTEM_PROMPT}]
        for h in reversed(history):
            messages.append({"role": "user", "content": h['user_message']})
            messages.append({"role": "assistant", "content": h['bot_response']})
        messages.append({"role": "user", "content": message})
        
        # Call OpenRouter API
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
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            bot_response = result['choices'][0]['message']['content']
            
            # Save to history
            if user_id:
                conn = get_db()
                c = conn.cursor()
                c.execute('INSERT INTO chat_history (user_id, user_message, bot_response) VALUES (?, ?, ?)',
                          (user_id, message, bot_response))
                conn.commit()
                conn.close()
            
            return jsonify({"response": bot_response})
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return jsonify({"response": "I'm having trouble responding right now. Please try again."})
            
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": "I'm experiencing technical difficulties. Please try again later."})

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
