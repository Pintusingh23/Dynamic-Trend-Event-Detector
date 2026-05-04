# app.py
# Dynamic Trend & Event Detector — Live Demo

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import re
import nltk

# ✅ FIX: download safely (no blank screen)
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk()

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Trend & Event Detector",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Dynamic Trend & Event Detector")

st.markdown("---")

# ─────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(w)
             for w in text.split()
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ─────────────────────────────────────────
# BALANCED SAMPLE DATA ✅
# ─────────────────────────────────────────
sample_data = {
    'headline': [
        # Politics
        "Trump wins presidential election",
        "Biden signs new law",
        "Congress passes bill",
        "Senate election results announced",

        # Health
        "COVID cases rising globally",
        "New vaccine approved",
        "Hospitals report increase in patients",
        "Doctors warn about new virus",

        # Environment
        "Climate change affects global weather",
        "Carbon emissions reach new high",
        "UN climate summit agreement",
        "Global warming concerns rise",

        # Sports
        "Lakers win NBA championship",
        "India wins cricket world cup",
        "Olympics opening ceremony highlights",
        "Football team wins final match",

        # Economy
        "Stock market crashes",
        "Inflation rises in economy",
        "Tesla reports record profits",
        "Federal reserve increases rates"
    ]
}

df = pd.DataFrame(sample_data)
df['clean'] = df['headline'].apply(clean_text)

# ─────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────
@st.cache_resource
def train_lda():
    vectorizer = CountVectorizer(max_features=500)
    dtm = vectorizer.fit_transform(df['clean'])

    lda = LatentDirichletAllocation(
        n_components=5,
        random_state=42,
        max_iter=100
    )
    lda.fit(dtm)
    return lda, vectorizer

lda, vectorizer = train_lda()

topic_names = {
    0: "🏛️ Politics",
    1: "🏥 Health",
    2: "🌍 Environment",
    3: "⚽ Sports",
    4: "💰 Economy"
}

def predict_topic(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    topic_dist = lda.transform(vec)[0]
    return np.argmax(topic_dist), topic_dist

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.header("🔍 Enter Headline")
    user_input = st.text_area("Type here:")

    if st.button("Detect Topic"):
        if user_input:
            topic_id, dist = predict_topic(user_input)
            st.success(f"Detected: {topic_names[topic_id]}")
        else:
            st.warning("Enter something!")


