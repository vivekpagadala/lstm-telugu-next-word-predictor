from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = os.path.join("saved_weights", "my_saved_weights_telugu_50epochs.h5")
model = load_model(MODEL_PATH)

# Load character mappings
chars = "abcdefghijklmnopqrstuvwxyz.,!?'\"-() "  # Define characters based on training data
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# Generate text function
def generate_text(seed_text, length=200):
    seq_length = 60  # Ensure same seq_length used during training
    generated = seed_text

    for _ in range(length):
        x_pred = np.zeros((1, seq_length, len(chars)))
        
        for t, char in enumerate(seed_text):
            if char in char_to_int:
                x_pred[0, t, char_to_int[char]] = 1.0
        
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = int_to_char[next_index]

        generated += next_char
        seed_text = seed_text[1:] + next_char  # Shift input

    return generated

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# API Route for text generation
@app.route('/generate', methods=['POST'])
def generate():
    seed_text = request.json.get('seed_text', '')
    if len(seed_text) < 60:
        return jsonify({"error": "Seed text must be at least 60 characters"}), 400
    
    generated_text = generate_text(seed_text)
    return jsonify({"generated_text": generated_text})

if __name__ == '_main_':
    app.run(debug=True)