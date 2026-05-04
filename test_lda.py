import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
data = pd.read_csv('../data/clean_data.csv')
data.shape
docs_clean = data['clean_text'].fillna('').tolist()
docs_tokens = [d.split() for d in docs_clean]
len(docs_clean)
# Cell 4 - TF-IDF
vectorizer = CountVectorizer(max_features=10000, min_df=5, max_df=0.95)
dtm = vectorizer.fit_transform(docs_clean)
vocab = vectorizer.get_feature_names_out()
dtm.shape
lda = LatentDirichletAllocation(n_components=20, random_state=42,
                                 max_iter=20, learning_method='online')
lda.fit(dtm)
for i, topic in enumerate(lda.components_[:10]):
    top_words = [vocab[j] for j in topic.argsort()[-8:][::-1]]
    print(f"Topic {i}: {', '.join(top_words)}")
!pip install gensim -q
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary

lda_topics = [[vocab[j] for j in t.argsort()[-10:][::-1]]
              for t in lda.components_]

dictionary = Dictionary(docs_tokens)

lda_cm = CoherenceModel(topics=lda_topics, texts=docs_tokens,
                        dictionary=dictionary, coherence='c_v')
LDA_SCORE = lda_cm.get_coherence()
LDA_PERPLEXITY = lda.perplexity(dtm)

print(f'Coherence Score: {LDA_SCORE:.4f}')
print(f'Perplexity Score: {LDA_PERPLEXITY:.4f}')

# Cell 9 - LDA Graph
import matplotlib.pyplot as plt
import os
os.makedirs('../graphs', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx in range(6):
    topic = lda.components_[idx]
    top_idx = topic.argsort()[-10:][::-1]
    top_w = [vocab[i] for i in top_idx]
    top_v = [topic[i] for i in top_idx]
    axes[idx].barh(top_w[::-1], top_v[::-1], color='steelblue')
    axes[idx].set_title(f'Topic {idx}')

plt.suptitle(f'LDA Topics — Coherence: {LDA_SCORE:.4f} | Perplexity: {LDA_PERPLEXITY:.4f}')
plt.tight_layout()
plt.savefig('../graphs/lda_topics.png')
plt.show()