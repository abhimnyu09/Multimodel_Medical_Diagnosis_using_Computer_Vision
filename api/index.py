import os
import json
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
from transformers import BertTokenizer, TFBertModel
import requests 
from huggingface_hub import hf_hub_download

# --- Configuration ---
app = Flask(__name__)

# --- API and Model Repository Configuration ---
# IMPORTANT: Replace these placeholder values with your actual details.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "YOUR_OPENROUTER_KEY_HERE") 
HF_USERNAME = "abhimnyu09"
REPO_NAME = "derm-detect-multimodal-model"

# --- Model Loading from Hugging Face Hub ---
# This section downloads the model when the Vercel instance starts.
CUSTOM_OBJECTS = {'TFBertModel': TFBertModel} 
model = None
index_to_label = None

try:
    print("Downloading model and label mapping from Hugging Face Hub...")
    
    # Download the model file and get its local path
    model_path = hf_hub_download(repo_id=f"{HF_USERNAME}/{REPO_NAME}", filename="multimodal_derm_model.h5")
    print(f"Model downloaded to: {model_path}")
    
    # Load the model from the downloaded file
    with tf.keras.utils.custom_object_scope(CUSTOM_OBJECTS):
        model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully from Hub!")
    
    # Download the label mapping file and get its local path
    label_mapping_path = hf_hub_download(repo_id=f"{HF_USERNAME}/{REPO_NAME}", filename="label_mapping.json")
    print(f"Label mapping downloaded to: {label_mapping_path}")

    # Load the label mapping from the downloaded file
    with open(label_mapping_path, 'r') as f:
        label_mapping = json.load(f)
    index_to_label = {int(v): k for k, v in label_mapping.items()} # Ensure keys are integers
    print("✅ Label mapping loaded successfully from Hub!")

except Exception as e:
    print(f"Fatal error loading model or mapping from Hub: {e}")
    # Set model to None so the app reports an error instead of crashing
    model = None 

# --- Tokenizer Loading ---
# This is small and can be loaded directly.
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')

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
                "HTTP-Referer": "http://your-vercel-app-url.vercel.app", # Recommended to change this later
                "X-Title": "DermDetect AI",
            },
            data=json.dumps({
                "model": "mistralai/mistral-7b-instruct", 
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )
        response.raise_for_status()
        
        api_response = response.json()
        return api_response['choices'][0]['message']['content'].strip()

    except Exception as e:
        print(f"OpenRouter API error: {e}")
        return "Could not retrieve AI-generated recommendation. Please consult a healthcare professional."

# --- Flask Routes ---
@app.route('/')
def index_route(): # Renamed to avoid conflict with file name `index.py`
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or index_to_label is None:
        return jsonify({'error': 'Model or essential data not loaded. Please check server logs.'}), 500
    if 'image' not in request.files or 'symptoms' not in request.form:
        return jsonify({'error': 'Missing image or symptoms data'}), 400

    image_file = request.files['image']
    symptoms_text = request.form['symptoms']

    image_data = preprocess_image(image_file)
    text_data = preprocess_text(symptoms_text)

    predictions = model.predict([image_data, text_data['input_ids'], text_data['attention_mask']])
    
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    predicted_class_name = index_to_label.get(predicted_index, "Unknown") # Use .get for safety

    recommendation = get_recommendation(predicted_class_name)

    return jsonify({
        'predicted_class': predicted_class_name.upper(),
        'confidence': f"{confidence:.2%}",
        'recommendation': recommendation
    })

# This part is mainly for local testing and is not used by Vercel
if __name__ == '__main__':
    app.run(debug=True)
