from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

# TF-IDF
def get_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

# BERT embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_bert_embedding(texts):
    embeddings = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        
        # Mean pooling
        emb = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(emb[0])
        
    return embeddings