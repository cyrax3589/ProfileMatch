import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
def load_data():
    users = pd.read_csv("data/users.csv")
    feedback = pd.read_csv("data/feedback.csv")
    return users, feedback


# Combine text fields
def prepare_text(users):
    users["combined_text"] = (
        users["professional_summary"].fillna("") + " " +
        users["about_me"].fillna("")
    )
    return users


# Create TF-IDF matrix
def vectorize_text(users):
    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(
        users["combined_text"]
    )

    return tfidf_matrix


# Create similarity matrix
def create_similarity_matrix(tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix