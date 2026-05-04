# app.py
# Dynamic Trend & Event Detector — Real Demo

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
import os

# Download NLTK data securely
@st.cache_resource
def download_nltk():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

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

st.title("📰 Dynamic Trend & Event Detector (Real Demo)")
st.markdown("This demo trains a topic model on a sample of the **real news dataset** and predicts the topic of any headline you enter.")
st.markdown("---")

# ─────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_input_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ─────────────────────────────────────────
# LOAD REAL DATA & TRAIN MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_data_and_train_model():
    # Load a sample of the real dataset to keep training fast for the demo
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'clean_data.csv')
    df = pd.read_csv(data_path)
    
    # We sample 5000 records to ensure the Streamlit app loads reasonably fast
    # but still provides a "real" model rather than dummy data.
    df_sample = df.sample(n=5000, random_state=42).copy()
    
    # Drop NaNs in clean_text
    docs = df_sample['clean_text'].fillna('').tolist()
    
    # Vectorize
    vectorizer = CountVectorizer(max_features=2000, min_df=5, max_df=0.9)
    dtm = vectorizer.fit_transform(docs)
    
    # Train LDA
    lda = LatentDirichletAllocation(
        n_components=10, # 10 distinct topics for the demo
        random_state=42,
        max_iter=15
    )
    lda.fit(dtm)
    
    # Extract top words for each topic to generate a name
    vocab = vectorizer.get_feature_names_out()
    topic_names = {}
    for i, topic in enumerate(lda.components_):
        top_words = [vocab[j] for j in topic.argsort()[-5:][::-1]]
        topic_names[i] = f"Topic {i+1}: " + ", ".join(top_words).title()
        
    return lda, vectorizer, topic_names

with st.spinner("Loading real data and training LDA model... This may take a minute on the first run."):
    lda_model, vectorizer, topic_names = load_data_and_train_model()

def predict_topic(text):
    clean = clean_input_text(text)
    # If the user input is completely filtered out by stopwords, handle gracefully
    if not clean.strip():
        return None, None
    vec = vectorizer.transform([clean])
    topic_dist = lda_model.transform(vec)[0]
    best_topic_id = np.argmax(topic_dist)
    return best_topic_id, topic_dist

# ─────────────────────────────────────────
# UI
# ─────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Enter a News Headline or Article Snippet")
    user_input = st.text_area("Type here:", placeholder="e.g., The stock market saw a massive crash today as tech companies...", height=150)

    if st.button("Detect Topic", type="primary"):
        if user_input.strip():
            topic_id, dist = predict_topic(user_input)
            if topic_id is not None:
                st.success(f"**Detected Topic:** {topic_names[topic_id]}")
                
                # Show confidence
                confidence = dist[topic_id] * 100
                st.info(f"Confidence: {confidence:.1f}%")
            else:
                st.warning("The text you entered didn't contain enough meaningful keywords (or only contained stop words) to detect a topic. Please provide more context.")
        else:
            st.warning("Please enter some text to analyze!")

with col2:
    st.header("📚 Discovered Topics")
    st.markdown("The model dynamically discovered these 10 topics from the real dataset:")
    for t_id, t_name in topic_names.items():
        st.markdown(f"- **{t_name}**")
