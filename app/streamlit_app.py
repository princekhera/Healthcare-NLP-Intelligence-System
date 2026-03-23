import streamlit as st
import torch
import joblib
import os
import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import nltk
nltk.download('stopwords')


from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------
# Load model + tokenizer
# -------------------------
@st.cache_resource
def load_model():
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "bert_model")
    ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
    
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    
    return model, tokenizer, le, device


model, tokenizer, le, device = load_model()


# -------------------------
# Prediction function
# -------------------------
def predict(text):
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    
    # 🔥 TOP 3
    topk = torch.topk(probs, 3)
    
    results = []
    
    for i in range(3):
        class_idx = topk.indices[0][i].item()
        confidence = topk.values[0][i].item()
        label = le.inverse_transform([class_idx])[0]
        
        results.append((label, confidence))
    
    return results

def explain_prediction(text):
    
    import re
    
    clean_words = re.findall(r'\b\w+\b', text)
    
    base_results = predict(text)
    base_confidence = base_results[0][1]
    
    word_importance = []
    
    for i, word in enumerate(clean_words):
        
        # 🚫 Skip useless words
        if word.lower() in stop_words:
            continue
        
        modified_words = clean_words[:i] + clean_words[i+1:]
        modified_text = " ".join(modified_words)
        
        results = predict(modified_text)
        new_confidence = results[0][1]
        
        importance = base_confidence - new_confidence
        
        word_importance.append((word.lower(), importance))
    
    return word_importance

# -------------------------
# UI
# -------------------------
st.title("Healthcare NLP Intelligence System")

st.write("Enter a medical abstract or query to classify disease category")

user_input = st.text_area("Enter text here:")

if st.button("Predict"):
    importance = explain_prediction(user_input)

    # Sort by importance
    importance = sorted(importance, key=lambda x: x[1], reverse=True)
    
    if user_input.strip() == "":

        st.warning("Please enter some text.")
    else:
        results = predict(user_input)

        # Top prediction
        st.success(f"Top Prediction: {results[0][0]}")
        st.info(f"Confidence: {results[0][1]:.2f}")

        st.write("### Top 3 Predictions")

        for label, confidence in results:
            st.write(label)
            st.progress(float(confidence))
        
        st.write("### Explanation (Important Words)")

        importance = explain_prediction(user_input)

        importance = sorted(importance, key=lambda x: x[1], reverse=True)

        top_words = set([w for w, _ in importance[:5]])

        # Boost key medical words
        domain_keywords = ["lung", "cancer", "diabetes", "pneumonia", "cardiac"]

        for word in user_input.lower().split():
            if word in domain_keywords:
                top_words.add(word)

        highlighted_text = ""


        for word in user_input.split():
            clean_word = re.sub(r'\W+', '', word).lower()
    
            if clean_word in top_words:
                highlighted_text += f"**:orange[{word}]** "
            else:
                highlighted_text += word + " "

        st.markdown(highlighted_text)