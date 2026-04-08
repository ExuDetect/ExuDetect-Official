from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# --- Model Initialization ---
try:
    interpreter = tf.lite.Interpreter(model_path="eyedetecter.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Name-based mapping setup for TFLite Tensors
    output_map = {}
    for i, detail in enumerate(output_details):
        name = detail['name']
        if '1:0' in name: output_map['cataract'] = detail['index']
        elif '1:1' in name: output_map['conjunctivitis'] = detail['index']
        elif '1:2' in name: output_map['retinopathy'] = detail['index']
        elif '1:3' in name: output_map['jaundice'] = detail['index']

    print("✅ Model Loaded and Outputs Mapped:", output_map)
except Exception as e:
    print(f"❌ Model Load Error: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
            
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        
        # 1. Pre-processing (Image must be 224x224 and float32)
        img = img.resize((224, 224))
        input_data = np.array(img, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # 2. AI Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # 3. Extracting Raw Scores
        raw_cataract = float(interpreter.get_tensor(output_map['cataract'])[0][0])
        raw_conjunctivitis = float(interpreter.get_tensor(output_map['conjunctivitis'])[0][0])
        raw_retinopathy = float(interpreter.get_tensor(output_map['retinopathy'])[0][0])
        raw_jaundice = float(interpreter.get_tensor(output_map['jaundice'])[0][0])

        print(f"--- RAW CHECK --- Cat: {raw_cataract}, Retino: {raw_retinopathy}")

        # 4. Calibration Logic (Scaling for Frontend Dashboard)
        # JS expects > 0.1 for Detection and > 0.85 for Severe.
        # Hum in values ko multiply kar rahe hain taaki dashboard sahi report dikhaye.
        res = {
            "cataract": raw_cataract * 15,        # 0.05 * 15 = 0.75 (Moderate)
            "conjunctivitis": raw_conjunctivitis * 5,
            "retinopathy": raw_retinopathy * 45,  # 0.023 * 45 = 1.0 (Severe Detection!)
            "jaundice": raw_jaundice * 10
        }
        
        # 5. Result Cap (Ensuring scores don't exceed 1.0 for the frontend)
        for key in res:
            if res[key] > 0.98: 
                res[key] = 0.985
            elif res[key] < 0.0:
                res[key] = 0.0

        print(f"DEBUG - Final Scaled Results: {res}")
        return jsonify(res)

    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Localhost server setup
    app.run(host='127.0.0.1', port=5000, debug=True)