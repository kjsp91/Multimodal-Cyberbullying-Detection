# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 20:18:48 2024

@author: kdkcs
"""
'''
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image

app = Flask(__name__)

# Path to the saved model
MODEL_PATH = 'FINAL VGG16.keras'
model = tf.keras.models.load_model(MODEL_PATH)

#path to tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 

def preprocess_image(image_path):
    """Preprocess image for prediction."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text.strip()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, prediction, and text extraction."""
    if 'file' not in request.files: 
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    # Image classification
    img_array = preprocess_image(filepath)
    prediction = model.predict(img_array)
    label = 'Non-Bully' if prediction[0] >= 0.5 else 'Bully'

    # Text extraction using OCR
    extracted_text = extract_text_from_image(filepath)

    # Clean up uploaded file
    os.remove(filepath)

    return jsonify({'prediction': label, 'extracted_text': extracted_text})

if __name__ == '__main__':
    app.run(debug=True)
'''
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename
import pytesseract
from PIL import Image
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel

app = Flask(__name__)

# --------- Image Classification Setup ---------
# Path to the saved image classification model
IMAGE_MODEL_PATH = 'FINAL VGG16.keras'
image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Preprocess image for prediction."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def extract_text_from_image(image_path):
    """Extract text from an image using Tesseract OCR."""
    img = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(img)
    return extracted_text.strip()

# --------- Text Classification Setup ---------
# Define the CustomXLMGRUTransformer class
class CustomXLMGRUTransformer(nn.Module):
    def __init__(self, hidden_size=512, num_labels=2, dropout_rate=0.5):
        super(CustomXLMGRUTransformer, self).__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')

        # BiGRU layer
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        # Transformer layer
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_size * 2, nhead=8)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids_xlm, attention_mask_xlm):
        xlm_outputs = self.xlm_roberta(input_ids_xlm, attention_mask=attention_mask_xlm)
        sequence_output = xlm_outputs.last_hidden_state

        gru_outputs, _ = self.gru(sequence_output)
        transformer_output = self.transformer_layer(gru_outputs)
        pooled_output = torch.mean(transformer_output, dim=1)
        normalized_output = self.batch_norm(pooled_output)

        combined = self.dropout(normalized_output)
        combined = torch.relu(self.fc1(combined))
        logits = self.fc2(combined)

        return logits

# Load text classification model
TEXT_MODEL_PATH = 'FINAL XLM.pth'
text_model = CustomXLMGRUTransformer()
text_model = torch.load(TEXT_MODEL_PATH, map_location=torch.device('cpu'))
text_model.eval()

# Load tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

def classify_text(text):
    """Classify text as Bully or Non-Bully."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=50)
    with torch.no_grad():
        logits = text_model(inputs['input_ids'], inputs['attention_mask'])
        predicted_class = torch.argmax(logits, dim=1).item()
    return "Bully" if predicted_class == 1 else "Non-Bully"

# --------- Flask Routes ---------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload, prediction, and text classification."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    # Image classification
    img_array = preprocess_image(filepath)
    prediction = image_model.predict(img_array)
    image_label = 'Non-Bully' if prediction[0] >= 0.5 else 'Bully'
 
    # Text extraction
    extracted_text = extract_text_from_image(filepath)

    # Text classification
    if extracted_text:
        text_label = classify_text(extracted_text)
    else:
        text_label = "No text extracted"
 
    # Clean up uploaded file
    os.remove(filepath)
    
    #return jsonify({'image_prediction': image_label, 'extracted_text': extracted_text, 'text_prediction': text_label})


    return jsonify({'image_prediction': image_label, 'text_prediction': text_label})

if __name__ == '__main__':
    app.run(debug=True)
