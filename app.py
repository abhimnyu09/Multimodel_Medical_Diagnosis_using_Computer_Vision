import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
from transformers import BertTokenizer, TFBertModel
import requests # <--- 1. Import 'requests' instead of 'openai'

# --- Configuration and Model Loading ---
app = Flask(__name__)

# <--- 2. Set your OPENROUTER API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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") 

# Load the trained multimodal model
MODEL_PATH = 'multimodal_derm_model.h5'
CUSTOM_OBJECTS = {'TFBertModel': TFBertModel} 

try:
    with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the tokenizer
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

# Load label mapping
with open('label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
    index_to_label = {v: k for k, v in label_mapping.items()}

# --- Helper Functions ---
def preprocess_image(image_file):
    img = Image.open(image_file.stream).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_text(text):
    return TOKENIZER(
        text,
        padding='max_length',
        truncation=True,
        max_length=64,
        return_tensors='tf'
    )

# <--- 3. This ENTIRE function is replaced to use OpenRouter ---
def get_recommendation(predicted_class):
    prompt = f"""
    An AI model has predicted a skin condition as '{predicted_class}'.
    Provide a brief, simple explanation of what this condition generally is, common characteristics, and what the typical next steps might be.
    IMPORTANT: Start your response with a clear disclaimer that this is NOT medical advice and a consultation with a qualified dermatologist is essential for any real diagnosis.
    Do not use alarming language. Keep the tone calm and informative.
    """
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                # These headers are recommended by OpenRouter
                "HTTP-Referer": "http://localhost:5000", 
                "X-Title": "DermDetect AI",
            },
            data=json.dumps({
                # Note the model name format is 'vendor/model'
                # Mistral 7B is a great, free model on OpenRouter
                "model": "mistralai/mistral-7b-instruct", 
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        response.raise_for_status() # Will raise an exception for HTTP errors
        
        # The response structure is the same as OpenAI's
        api_response = response.json()
        return api_response['choices'][0]['message']['content'].strip()

    except Exception as e:
        print(f"OpenRouter API error: {e}")
        return "Could not retrieve AI-generated recommendation. Please consult a healthcare professional."

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    if 'image' not in request.files or 'symptoms' not in request.form:
        return jsonify({'error': 'Missing image or symptoms data'}), 400

    image_file = request.files['image']
    symptoms_text = request.form['symptoms']

    image_data = preprocess_image(image_file)
    text_data = preprocess_text(symptoms_text)

    predictions = model.predict([image_data, text_data['input_ids'], text_data['attention_mask']])
    
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_class_name = index_to_label[predicted_index]

    recommendation = get_recommendation(predicted_class_name)

    return jsonify({
        'predicted_class': predicted_class_name.upper(),
        'confidence': f"{confidence:.2%}",
        'recommendation': recommendation
    })

if __name__ == '__main__':
    app.run(debug=True)
