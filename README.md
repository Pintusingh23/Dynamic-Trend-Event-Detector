# Dynamic Trend & Event Detector
## Emerging Topic Detection & News Correlation

**Course:** Deep Learning & Advanced Machine Learning — 6th Semester

## Authors
Pintu Singh (230105)
Fathal (230043)

---

## Project Overview

This project detects emerging topics from large-scale news articles 
and analyzes whether a trend corresponds to a real-world event 
or a short-lived viral topic.

We use topic modeling and temporal analysis to track how 
discussions evolve over time. We compare a simple frequency-based 
baseline with a probabilistic topic model (LDA) to identify 
meaningful patterns in news streams.

---

## Problem Statement

Massive volumes of news articles are generated every day, 
making it difficult to identify important events.

Traditional approaches fail to capture:
- Temporal dynamics of topics
- Semantic evolution of discussions
- Difference between real events and viral trends

We propose a system that:
- Detects emerging topics automatically
- Tracks topic evolution over time
- Differentiates meaningful events from temporary trends

---

## Methodology

1. Data collection from news dataset
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Baseline: Simple Word Frequency extraction
5. Advanced ML: Topic modeling using LDA
6. Temporal topic tracking
7. Model comparison and evaluation

---

## Results

| Model | Type | Coherence Score | Temporal |
|-------|------|----------------|----------|
| Baseline | Word Frequency | — | No |
| Model A — LDA | Advanced ML | 0.3573 | No |

---

## File Structure
```
Dynamic-Trend-Event-Detector/
├── data/
│   └── clean_data.csv
├── graphs/
│   ├── eda1_categories.png
│   ├── eda2_yearly.png
│   ├── eda3_wordcount.png
│   ├── eda4_topwords.png
│   └── lda_topics.png
├── notebooks/
│   ├── clean.ipynb
│   ├── descriptive_analysis.ipynb
│   ├── eda_plan.ipynb
│   └── predictive_analysis.ipynb
├── reports/
│   └── ablation_table.csv
└── README.md
```

---

## Dataset

- **Source:** Kaggle — News Category Dataset
- **Link:** https://www.kaggle.com/datasets/rmisra/news-category-dataset
- **Size:** 209,527 articles (2012–2022)
- **Sampled:** 10,000 articles

---

## How To Run
```bash
# Step 1 - Install libraries
pip install pandas numpy scikit-learn 
pip install matplotlib seaborn gensim nltk

# Step 2 - Run notebooks in order
1. notebooks/clean.ipynb
2. notebooks/descriptive_analysis.ipynb
3. notebooks/eda_plan.ipynb
4. notebooks/predictive_analysis.ipynb
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| CountVectorizer (sklearn) | TF-IDF features |
| LDA (sklearn) | Advanced ML topic modeling |
| Gensim | Coherence score evaluation |
| NLTK | Text preprocessing |
| Matplotlib | Visualizations |

---

## References

1. Blei et al. (2003) — Latent Dirichlet Allocation. JMLR.
2. Misra (2022) — News Category Dataset. Kaggle.