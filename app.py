from flask import Flask, render_template, request, jsonify
import pickle
import json
import os
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables
model = None
vectorizer = None
config = None

def load_model_and_config():
    """Load sklearn model and configuration"""
    global model, vectorizer, config
    
    print("Loading model and configuration...")
    
    # Load config
    with open('models/config.json', 'r') as f:
        config = json.load(f)
    
    # Load model and vectorizer
    with open('models/model_sklearn.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/vectorizer_sklearn.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("✅ Model loaded successfully!")
    print(f"Model accuracy: {config['accuracy'] * 100:.2f}%")

def predict_dialogue_act(text):
    """Predict dialogue act for text"""
    # Vectorize
    text_vec = vectorizer.transform([text])
    
    # Predict
    predicted_class = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    # Get label
    predicted_label = config['id2label'][str(predicted_class)]
    confidence = probabilities[predicted_class]
    
    # All probabilities
    all_probs = {
        config['id2label'][str(i)]: float(probabilities[i])
        for i in range(len(config['id2label']))
    }
    
    return predicted_label, confidence, all_probs

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
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        response = {
            'success': True,
            'text': text,
            'predicted_label': predicted_label,
            'confidence': round(confidence * 100, 2),
            'all_probabilities': {k: round(v * 100, 2) for k, v in all_probs.items()},
            'sorted_probabilities': [
                {'label': label, 'probability': round(prob * 100, 2)}
                for label, prob in sorted_probs
            ]
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-batch', methods=['POST'])
def api_predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        
        if 'text' not in df.columns:
            return jsonify({'error': 'CSV must have a "text" column'}), 400
        
        results = []
        for idx, row in df.iterrows():
            text = str(row['text']).strip()
            if text:
                predicted_label, confidence, _ = predict_dialogue_act(text)
                results.append({
                    'text': text,
                    'predicted_label': predicted_label,
                    'confidence': round(confidence * 100, 2)
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
        
        response = {
            'model_type': config['model_type'],
            'num_classes': config['num_labels'],
            'classes': list(config['id2label'].values()),
            'overall_accuracy': round(config['accuracy'] * 100, 2),
            'training_date': config['training_date'],
            'class_metrics': class_metrics
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/examples', methods=['GET'])
def api_examples():
    examples = {
        "Greeting": [
            "Hello! How can I help you today?",
            "Good morning! Welcome to our support center.",
            "Hi there! Thanks for reaching out to us."
        ],
        "Question": [
            "What is your order number?",
            "Can you provide your account email address?",
            "When did you first notice this issue?"
        ],
        "Answer": [
            "Your order was shipped on December 20th.",
            "The refund will be processed within 5-7 business days.",
            "Your account has been successfully verified."
        ],
        "Complaint": [
            "I've been waiting for my order for 3 weeks now!",
            "This is the third time I'm contacting support about this issue.",
            "The product I received is completely different from what I ordered."
        ],
        "Request": [
            "I would like to cancel my subscription please.",
            "Can you please expedite the shipping on my order?",
            "I need to update my billing address."
        ],
        "Acknowledgment": [
            "Thank you for the information.",
            "Okay, I understand now.",
            "Got it, thanks for clarifying."
        ],
        "Closing": [
            "Thank you for your help! Have a great day.",
            "Thanks, that solved my issue. Goodbye!",
            "I appreciate your assistance. Take care!"
        ]
    }
    
    return jsonify(examples)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    load_model_and_config()
    
    print("\n" + "=" * 80)
    print(" " * 20 + "🚀 DIALOGUE ACT CLASSIFIER - WEB APPLICATION")
    print("=" * 80)
    print("\n✅ Server starting...")
    print("📡 Open your browser and go to: http://127.0.0.1:5000")
    print("\nPress CTRL+C to stop the server")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)