import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class DialogueActDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class for dialogue act classification
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_and_preprocess_data():
    """
    Load the dataset and prepare for training
    """
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv("data/customer_support_dialogues.csv")
    
    print(f"   Total samples loaded: {len(df)}")
    
    # Create label mapping
    unique_labels = sorted(df['dialogue_act'].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    # Convert labels to integers
    df['label_id'] = df['dialogue_act'].map(label2id)
    
    print(f"   Dialogue acts: {unique_labels}")
    
    return df, label2id, id2label

def create_train_test_split(df, test_size=0.2):
    """
    Split data into training and testing sets
    """
    print(f"\n[2/6] Splitting data (train: {int((1-test_size)*100)}%, test: {int(test_size*100)}%)...")
    
    X = df['text'].values
    y = df['label_id'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def prepare_datasets(X_train, X_test, y_train, y_test, tokenizer):
    """
    Create PyTorch datasets
    """
    print("\n[3/6] Preparing datasets and tokenizing...")
    
    train_dataset = DialogueActDataset(X_train, y_train, tokenizer)
    test_dataset = DialogueActDataset(X_test, y_test, tokenizer)
    
    print(f"   Train dataset ready: {len(train_dataset)} samples")
    print(f"   Test dataset ready: {len(test_dataset)} samples")
    
    return train_dataset, test_dataset

def train_model(train_dataset, test_dataset, num_labels, id2label, label2id):
    """
    Train DistilBERT model
    """
    print("\n[4/6] Training DistilBERT model...")
    print("   (This may take 5-10 minutes depending on your hardware)")
    
    # Load pre-trained model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./models/checkpoints',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./models/logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        return {"accuracy": accuracy}
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("   Training started...")
    trainer.train()
    print("   ✅ Training complete!")
    
    return model, trainer

def evaluate_model(model, test_dataset, X_test, y_test, id2label, tokenizer):
    """
    Evaluate model and generate metrics
    """
    print("\n[5/6] Evaluating model performance...")
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for text in X_test:
            inputs = tokenizer(
                text,
                return_tensors='pt',
                max_length=128,
                padding='max_length',
                truncation=True
            )
            
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n   📊 Overall Accuracy: {accuracy * 100:.2f}%")
    
    # Classification report
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(y_test, predictions, target_names=target_names, output_dict=True)
    
    print("\n   Classification Report:")
    print("   " + "=" * 70)
    for label in target_names:
        metrics = report[label]
        print(f"   {label:20s} | Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
    print("   " + "=" * 70)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Dialogue Act Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs('models', exist_ok=True)
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n   ✅ Confusion matrix saved: models/confusion_matrix.png")
    
    return accuracy, report

def save_model_and_config(model, tokenizer, label2id, id2label, accuracy, report):
    """
    Save trained model and configuration
    """
    print("\n[6/6] Saving model and configuration...")
    
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model.save_pretrained('models/dialogue_act_classifier')
    tokenizer.save_pretrained('models/dialogue_act_classifier')
    
    # Save configuration
    config = {
        'model_type': 'DistilBERT',
        'num_labels': len(label2id),
        'label2id': label2id,
        'id2label': {int(k): v for k, v in id2label.items()},
        'accuracy': float(accuracy),
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'classification_report': report
    }
    
    with open('models/config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("   ✅ Model saved: models/dialogue_act_classifier/")
    print("   ✅ Config saved: models/config.json")

def main():
    """
    Main training pipeline
    """
    print("=" * 80)
    print(" " * 20 + "DIALOGUE ACT CLASSIFIER - MODEL TRAINING")
    print("=" * 80)
    
    # Load data
    df, label2id, id2label = load_and_preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = create_train_test_split(df)
    
    # Initialize tokenizer
    print("\n   Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(X_train, X_test, y_train, y_test, tokenizer)
    
    # Train model
    model, trainer = train_model(train_dataset, test_dataset, len(label2id), id2label, label2id)
    
    # Evaluate
    accuracy, report = evaluate_model(model, test_dataset, X_test, y_test, id2label, tokenizer)
    
    # Save everything
    save_model_and_config(model, tokenizer, label2id, id2label, accuracy, report)
    
    print("\n" + "=" * 80)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")
    print("\nYou can now run the web application using: python app.py")
    print("=" * 80)

if __name__ == "__main__":
    main()