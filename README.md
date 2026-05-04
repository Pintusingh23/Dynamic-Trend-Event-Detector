# Dynamic Trend & Event Detector

## Emerging Topic Detection & News Correlation

**Course:** Deep Learning & Advanced Machine Learning 
## Project

Dynamic Trend & Event Detector

## Authors

Pintu Singh (230105)
Fathal (230043)

---

## Project Overview

This project detects emerging topics from large-scale news articles and analyzes whether a trend corresponds to a real-world event or a short-lived viral topic.

The system uses topic modeling, semantic embeddings, and temporal analysis to track how discussions evolve over time. We compare traditional topic modeling (LDA), deep learning-based topic modeling (BERTopic), and a hybrid approach using semantic velocity and external news correlation.

The goal is to automatically identify meaningful events from noisy information streams and distinguish real-world events from temporary trends.

---

## Problem Statement

Massive volumes of news articles are generated every day, making it difficult to identify important events. Many trends appear suddenly and disappear quickly, while others represent real-world events such as elections, pandemics, or policy changes.

Traditional topic modeling approaches fail to capture temporal dynamics and semantic evolution of topics. They also cannot differentiate between meaningful events and short-lived viral trends.

Therefore, we propose a Dynamic Trend & Event Detector that:

* Detects emerging topics automatically
* Measures semantic velocity of topics
* Tracks topic evolution over time
* Correlates topics with real-world events
* Differentiates meaningful events from temporary trends

---

## Methodology

The system follows a hybrid pipeline:

1. Data collection from news dataset
2. Data cleaning and preprocessing
3. Exploratory data analysis (EDA)
4. Topic modeling using LDA (baseline)
5. Deep topic modeling using BERTopic
6. Temporal topic tracking
7. Semantic velocity calculation
8. External event correlation (GDELT)
9. Hybrid model comparison

---

## Results

| Model        | Type          | Coherence Score | Temporal | GDELT |
| ------------ | ------------- | --------------- | -------- | ----- |
| A — LDA      | Advanced ML   | 0.3573          | No       | No    |
| B — BERTopic | Deep Learning | 0.4781          | No       | No    |
| C — Hybrid   | Hybrid        | 0.4981          | Yes      | Yes   |

Hybrid system improved **39.4%** over LDA baseline.

---

## File Structure

```
project22/
├── data/
│   └── clean_data.csv
├── graphs/
│   ├── eda1_categories.png
│   ├── eda2_yearly.png
│   ├── eda3_wordcount.png
│   ├── eda4_topwords.png
│   ├── lda_topics.png
│   ├── topics_over_time.html
│   ├── semantic_velocity.png
│   └── ablation_graph.png
├── notebooks/
│   ├── clean.ipynb
│   ├── descriptive_analysis.ipynb
│   ├── eda_plan.ipynb
│   ├── predictive_analysis.ipynb
│   ├── modelBERTopic.ipynb
│   └── test1.ipynb
├── reports/
│   ├── ablation_table.csv
│   └── gdelt_results.csv
└── README.md
```

---

## Dataset

Source: Kaggle — News Category Dataset
Link: https://www.kaggle.com/datasets/rmisra/news-category-dataset
Size: 209,527 articles (2012–2022)

---

## How To Run

```bash
# Step 1 - Install libraries
pip install pandas numpy scikit-learn matplotlib seaborn
pip install bertopic sentence-transformers umap-learn hdbscan gensim

# Step 2 - Run notebooks in order
1. notebooks/clean.ipynb
2. notebooks/descriptive_analysis.ipynb
3. notebooks/eda_plan.ipynb
4. notebooks/predictive_analysis.ipynb
5. notebooks/modelBERTopic.ipynb
6. notebooks/test1.ipynb
```

---

## Tech Stack

| Tool          | Purpose                    |
| ------------- | -------------------------- |
| LDA (sklearn) | Advanced ML baseline       |
| SBERT         | Deep learning embeddings   |
| UMAP          | Dimensionality reduction   |
| HDBSCAN       | Density-based clustering   |
| BERTopic      | Neural topic modeling      |
| GDELT API     | External news verification |

---

## References

1. Blei et al. (2003) — Latent Dirichlet Allocation
2. Grootendorst (2022) — BERTopic
3. Reimers & Gurevych (2019) — Sentence-BERT
4. McInnes & Healy (2017) — HDBSCAN
5. Misra (2022) — News Category Dataset, Kaggle
