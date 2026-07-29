import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Store documents
docs = []

# Input number of documents
n = int(input("Enter number of documents: "))

# Input documents
for i in range(n):
    doc = input(f"Enter document {i+1}: ")
    docs.append(doc)

# Input search query
query = input("\nEnter search query: ")

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer()

# Convert documents into TF-IDF vectors
X = vectorizer.fit_transform(docs)

# Convert query into TF-IDF vector
query_vec = vectorizer.transform([query])

# Calculate cosine similarity
scores = cosine_similarity(query_vec, X)

print("\n========== TF-IDF Similarity Scores ==========")
for i, score in enumerate(scores[0]):
    print(f"Document {i+1}: {round(score, 3)}")

# -------------------------------
# LSA (Latent Semantic Analysis)
# -------------------------------
# Number of components should not exceed number of documents
components = min(2, len(docs))

svd = TruncatedSVD(n_components=components)

# Reduce document vectors
X_lsa = svd.fit_transform(X)

# Reduce query vector
query_lsa = svd.transform(query_vec)

# Compute similarity in semantic space
lsa_scores = cosine_similarity(query_lsa, X_lsa)

print("\n========== LSA Similarity Scores ==========")
for i, score in enumerate(lsa_scores[0]):
    print(f"Document {i+1}: {round(score, 3)}")

# Find best matching document
best = np.argmax(lsa_scores)

print("\n========== Most Relevant Document ==========")
print(docs[best])