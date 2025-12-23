import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

print("=" * 80)
print(" " * 15 + "DIALOGUE ACT CLASSIFIER - SKLEARN VERSION")
print("=" * 80)

# Load dataset
print("\n[1/5] Loading dataset...")
df = pd.read_csv("data/customer_support_dialogues.csv")
print(f"   Total samples: {len(df)}")

# Create label mapping
labels = sorted(df['dialogue_act'].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
print(f"   Dialogue acts: {labels}")

# Prepare data
X = df['text'].values
y = df['dialogue_act'].map(label2id).values

# Train-test split
print("\n[2/5] Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Vectorization
print("\n[3/5] Vectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(f"   Feature dimensions: {X_train_vec.shape[1]}")

# Training
print("\n[4/5] Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs', C=1.0)
model.fit(X_train_vec, y_train)
print("   ✅ Training complete!")

# Evaluation
print("\n[5/5] Evaluating model...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n   📊 Overall Accuracy: {accuracy * 100:.2f}%")
print("\n   Classification Report:")
print("   " + "=" * 70)

report_dict = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
for label in labels:
    metrics = report_dict[label]
    print(f"   {label:20s} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
print("   " + "=" * 70)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Dialogue Act Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Save
os.makedirs('models', exist_ok=True)
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n   ✅ Confusion matrix saved: models/confusion_matrix.png")

# Save model
with open('models/model_sklearn.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/vectorizer_sklearn.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save config
config = {
    'model_type': 'Logistic Regression (sklearn)',
    'accuracy': float(accuracy),
    'label2id': label2id,
    'id2label': id2label,
    'num_labels': len(labels),
    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'classification_report': report_dict
}

with open('models/config.json', 'w') as f:
    json.dump(config, f, indent=4)

print("\n   ✅ Model saved: models/model_sklearn.pkl")
print("   ✅ Vectorizer saved: models/vectorizer_sklearn.pkl")
print("   ✅ Config saved: models/config.json")

print("\n" + "=" * 80)
print("🎉 MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")
print("\nYou can now run: python app.py")
print("=" * 80)