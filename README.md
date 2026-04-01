

# 🧠 Healthcare NLP Intelligence System

An end-to-end NLP system for classifying medical text into disease categories using a fine-tuned BERT model, with an interactive Streamlit interface and built-in explainability.

---
<p align="center">
  <img src="healthcare_NLP_demo.gif" width="800">
</p>

## 🚀 Overview

This project focuses on solving a real healthcare problem:
making sense of unstructured medical text (like research abstracts or clinical descriptions).

Here’s what it does:

* Takes medical text as input (e.g., PubMed abstracts)
* Classifies it into disease categories
* Shows top predictions with confidence scores
* Highlights the most important words driving the prediction

What this really means is — it's not just prediction, it explains *why*.

---

## 🧠 Models Used

### 1. BERT (Final Model)

* Pretrained: `bert-base-uncased`
* Fine-tuned on medical abstracts dataset
* Handles context, semantics, and domain-specific language effectively

### 2. TF-IDF (Baseline - Initial Approach)

* Used during early experimentation
* Provided a benchmark before moving to transformer-based models
* Highlighted limitations in capturing contextual meaning

---

## 🏗️ Project Architecture

```
├── app/
│   └── streamlit_app.py        # Streamlit UI
├── models/
│   ├── bert_model/             # Fine-tuned BERT model
│   └── label_encoder.pkl       # Label encoder
├── data/
│   └── pubmed.csv              # Training dataset
├── training/
│   └── train.py                # Training pipeline
├── README.md
```

---

## ⚙️ Features

### ✅ Disease Classification

* Predicts top 3 disease categories
* Uses softmax probabilities for confidence scoring

### 📊 Explainability (Key Feature)

* Word importance via **perturbation-based analysis**
* Removes each word and measures drop in confidence
* Highlights most influential medical terms

### 🎯 Smart Highlighting

* Important words are visually emphasized in output
* Domain-specific keywords (e.g., lung, cancer, diabetes) are boosted

### 💻 Interactive UI

* Built with Streamlit
* Simple input → instant predictions → visual explanations

---


## 🧪 Training Pipeline

### Steps:

1. Load dataset (PubMed abstracts)
2. Encode labels using `LabelEncoder`
3. Train-validation split (stratified)
4. Tokenization using BERT tokenizer
5. Fine-tuning using Hugging Face `Trainer`

### Training Configuration:

* Epochs: 3
* Learning Rate: 2e-5
* Batch Size: 8
* Weight Decay: 0.01

### Evaluation Metrics:

* Accuracy
* Weighted F1 Score

---

## 🔍 Example Workflow

1. Input:

   ```
   Lung cancer is a leading cause of cancer-related deaths...
   ```

2. Output:

   * Top Prediction: Lung Cancer
   * Confidence: 0.92
   * Top 3 predictions with probability bars
   * Highlighted important words (e.g., lung, cancer)

---

## 🧠 Explainability Logic (Core Idea)

Instead of using complex attention visualization, this project uses a simple but powerful approach:

* Remove one word at a time
* Re-run prediction
* Measure confidence drop
* Higher drop = higher importance

This makes the model:

* More interpretable
* More trustworthy for healthcare use cases

---

## 🛠️ Installation

```bash
git clone https://github.com/princekhera/Healthcare-NLP-Intelligence-System.git
cd healthcare-nlp-intelligence-system
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app/streamlit_app.py
```

---

## 🏋️ Train the Model

```bash
python training/train.py
```

---

## 📦 Dependencies

* Python
* PyTorch
* Transformers (Hugging Face)
* Scikit-learn
* Pandas
* Streamlit
* NLTK

---

## ⚠️ Limitations

* Word importance method is computationally expensive (O(n) predictions per input)
* Not optimized for long clinical documents
* Limited domain keywords for boosting

---

## 🔮 Future Improvements

* Use SHAP / Integrated Gradients for better explainability
* Expand dataset (clinical notes, EHR data)
* Deploy as API (FastAPI + Docker)
* Improve UI with better visualization
* Multi-label classification (real-world scenario)

---

## 📌 Key Takeaways

* Transitioned from TF-IDF → BERT (major performance gain)
* Built full pipeline: training → inference → UI
* Added explainability (rare in beginner projects)
* Designed with real healthcare applications in mind
