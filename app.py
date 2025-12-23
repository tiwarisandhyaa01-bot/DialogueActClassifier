from flask import Flask, render_template, request, jsonify
import pickle
import json
import os
from werkzeug.utils import secure_filename
import pandas as pd

# -------------------- APP SETUP --------------------
app = Flask(
    __name__,
    template_folder='templates',
    static_folder='static',
    static_url_path='/static'
)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# -------------------- GLOBALS --------------------
model = None
vectorizer = None
config = None

# -------------------- LOAD MODEL --------------------
def load_model_and_config():
    global model, vectorizer, config

    with open('models/config.json', 'r') as f:
        config = json.load(f)

    with open('models/model_sklearn.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('models/vectorizer_sklearn.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    print("✅ Model & Vectorizer loaded successfully")

# -------------------- PREDICTION --------------------
def predict_dialogue_act(text):
    text_vec = vectorizer.transform([text])
    predicted_class = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]

    predicted_label = config['id2label'][str(predicted_class)]
    confidence = probabilities[predicted_class]

    all_probs = {
        config['id2label'][str(i)]: float(probabilities[i])
        for i in range(len(config['id2label']))
    }

    return predicted_label, confidence, all_probs

# -------------------- ROUTES --------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        predicted_label, confidence, all_probs = predict_dialogue_act(text)

        return jsonify({
            'success': True,
            'text': text,
            'predicted_label': predicted_label,
            'confidence': round(confidence * 100, 2),
            'all_probabilities': {
                k: round(v * 100, 2) for k, v in all_probs.items()
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-batch', methods=['POST'])
def api_predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'error': 'Invalid file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        df = pd.read_csv(filepath)
        if 'text' not in df.columns:
            return jsonify({'error': 'CSV must contain a text column'}), 400

        results = []
        for text in df['text'].astype(str):
            text = text.strip()
            if text:
                label, conf, _ = predict_dialogue_act(text)
                results.append({
                    'text': text,
                    'predicted_label': label,
                    'confidence': round(conf * 100, 2)
                })

        os.remove(filepath)

        return jsonify({
            'success': True,
            'total_predictions': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    try:
        report = config.get('classification_report', {})

        class_metrics = []
        for label in config['id2label'].values():
            if label in report:
                class_metrics.append({
                    'label': label,
                    'precision': round(report[label]['precision'] * 100, 2),
                    'recall': round(report[label]['recall'] * 100, 2),
                    'f1_score': round(report[label]['f1-score'] * 100, 2),
                    'support': report[label]['support']
                })

        return jsonify({
            'model_type': config['model_type'],
            'num_classes': config['num_labels'],
            'classes': list(config['id2label'].values()),
            'overall_accuracy': round(config['accuracy'] * 100, 2),
            'training_date': config['training_date'],
            'class_metrics': class_metrics
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/examples', methods=['GET'])
def api_examples():
    return jsonify({
        "Greeting": [
            "Hello! How can I help you today?",
            "Good morning! Welcome to our support center."
        ],
        "Question": [
            "What is your order number?",
            "Can you provide your account email?"
        ],
        "Complaint": [
            "I've been waiting for my order for weeks!",
            "The product is not what I ordered."
        ],
        "Request": [
            "I want to cancel my subscription.",
            "Please update my billing address."
        ],
        "Closing": [
            "Thanks for your help!",
            "Have a great day!"
        ]
    })

# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    load_model_and_config()

    port = int(os.environ.get("PORT", 8000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
