import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from datasets import Dataset


# -------------------------
# Load Data
# -------------------------
def load_data(path):
    df = pd.read_csv(path)
    df = df[['abstract', 'label']].dropna()
    return df


# -------------------------
# Encode Labels
# -------------------------
def encode_labels(df):
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    return df, le


# -------------------------
# Train/Test Split
# -------------------------
def split_data(df):
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label_encoded'],
        random_state=42
    )
    return train_df, val_df   # ✅ FIXED


# -------------------------
# Convert to Dataset
# -------------------------
def to_dataset(df):
    df = df.rename(columns={"label_encoded": "labels"})
    return Dataset.from_pandas(df[['abstract', 'labels']])


# -------------------------
# Tokenization
# -------------------------
def tokenize_data(dataset, tokenizer):
    def tokenize(example):
        return tokenizer(
            example['abstract'],
            truncation=True,
            padding='max_length',
            max_length=256
        )
    
    dataset = dataset.map(tokenize, batched=True)
    
    # Remove text column (cleaner)
    dataset = dataset.remove_columns(['abstract'])
    
    # Convert to PyTorch format
    dataset.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'labels']
    )
    
    return dataset


# -------------------------
# Model
# -------------------------
def get_model(num_labels):
    return BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )



from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }


# -------------------------
# Trainer
# -------------------------
def get_trainer(model, train_dataset, val_dataset):

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    return trainer


# -------------------------
# Main Pipeline
# -------------------------
def run_training():

    df = load_data("data/pubmed.csv")

    df, le = encode_labels(df)

    train_df, val_df = split_data(df)

    train_dataset = to_dataset(train_df)
    val_dataset = to_dataset(val_df)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = tokenize_data(train_dataset, tokenizer)
    val_dataset = tokenize_data(val_dataset, tokenizer)

    model = get_model(num_labels=len(le.classes_))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    trainer = get_trainer(model, train_dataset, val_dataset)

    trainer.train()

    results = trainer.evaluate()
    print(results)

    model.save_pretrained("models/bert_model")
    tokenizer.save_pretrained("models/bert_model")

    joblib.dump(le, "models/label_encoder.pkl")
    
    print("Training complete ✅")