import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def load_data(path):
    df = pd.read_csv(path)
    df = df[['abstract', 'label']]  # adjust columns if needed
    df['clean_text'] = df['abstract'].apply(clean_text)
    return df