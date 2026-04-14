<p align="center">
  <img 
    src="https://github.com/user-attachments/assets/842de2f2-aac4-480c-9d3b-268eb9d84139"
    width="105%"
    style="border-radius: 18px;"
  >
</p>
<h1 align="center">🔗 ProfileMatch</h1>
<h3 align="center">Intelligent Profile Matching & Recommendation System</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/NLP-NLTK%20%7C%20TF--IDF-009688?style=for-the-badge&logo=apache&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" />
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#system-architecture">Architecture</a> •
  <a href="#technology-stack">Stack</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#performance-metrics">Metrics</a> •
</p>

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Technology Stack](#technology-stack)
5. [Machine Learning Pipeline](#machine-learning-pipeline)
6. [Dataset](#dataset)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Project Structure](#project-structure)
10. [Performance Metrics](#performance-metrics)
11. [Use Cases](#use-cases)
12. [Future Enhancements](#future-enhancements)
13. [Contributing](#contributing)
14. [License](#license)
15. [Author](#author)

---

##  Overview

**ProfileMatch** is an end-to-end intelligent profile matching and recommendation system that leverages **Natural Language Processing** and **Machine Learning** to identify and rank the most compatible user profiles. By combining TF-IDF vectorization, cosine similarity scoring, and a logistic regression feedback model, ProfileMatch delivers high-accuracy recommendations in real time through an interactive Streamlit web interface.

Whether you're building a professional networking platform, a talent-matching engine, or a social discovery app — ProfileMatch provides a modular, production-ready ML backbone you can extend with confidence.

>  **Goal:** Bridge the gap between raw profile data and meaningful human connections using data-driven compatibility scoring.

---

## Features

| Feature | Description | Status |
|---|---|---|
|  **Intelligent Profile Similarity** | Analyzes text fields using TF-IDF vectors to compute semantic closeness between profiles | ✅ Live |
|  **Hybrid Compatibility Scoring** | Combines NLP similarity with structured attribute matching (age, location, profession) | ✅ Live |
|  **Natural Language Processing** | Processes `professional_summary` and `about_me` fields using NLTK tokenization and TF-IDF | ✅ Live |
|  **Real-Time Match Ranking** | Instantly ranks top-N compatible profiles for any query user | ✅ Live |
|  **Adaptive Learning** | Ingests user feedback (like/skip/connect) to retrain the logistic regression model | ✅ Live |
|  **Modular Architecture** | Cleanly separated preprocessing, modeling, scoring, and UI layers | ✅ Live |
|  **Interactive Web Interface** | Streamlit-powered dashboard for browsing matches and submitting feedback | ✅ Live |
|  **Visual Analytics** | Seaborn/Matplotlib charts for similarity distributions and match statistics | ✅ Live |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROFILEMATCH SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐        ┌──────────────────────────────────────┐
  │  Raw Data    │        │           DATA LAYER                 │
  │  users.csv   │──────▶│  Cleaning · Normalization · Merge     │
  │  feedback.csv│        │  (pandas / numpy)                    │
  └──────────────┘        └────────────────┬─────────────────────┘
                                           │
                          ┌────────────────▼─────────────────────┐
                          │         NLP PROCESSING               │
                          │  Tokenization · Stopword Removal     │
                          │  TF-IDF Vectorization (scikit-learn) │
                          └────────────────┬─────────────────────┘
                                           │
               ┌───────────────────────────▼──────────────────────────┐
               │                   ML PIPELINE                        │
               │                                                      │
               │  ┌─────────────────┐     ┌────────────────────────┐  │
               │  │Cosine Similarity│     │  Logistic Regression   │  │
               │  │ (Content Score) │     │  (Feedback Model)      │  │
               │  └────────┬────────┘     └───────────┬────────────┘  │
               │           │                          │               │
               │           └──────────┬───────────────┘               │
               │                      ▼                               │
               │           ┌──────────────────┐                       │
               │           │  Hybrid Scorer   │                       │
               │           │  (Weighted Rank) │                       │
               │           └──────────┬───────┘                       │
               └──────────────────────┼───────────────────────────────┘
                                      │
                          ┌───────────▼──────────────┐
                          │     STREAMLIT UI         │
                          │  Match Dashboard         │
                          │  Profile Viewer          │
                          │  Feedback Collector      │
                          │  Analytics Charts        │
                          └──────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | ![Python](https://img.shields.io/badge/-Python%203.10+-3776AB?logo=python&logoColor=white) | Core development language |
| **Data Processing** | `pandas`, `numpy` | Data ingestion, cleaning, and manipulation |
| **NLP** | `nltk`, `scikit-learn (TfidfVectorizer)` | Text preprocessing and vectorization |
| **ML Modeling** | `scikit-learn` | Cosine similarity, Logistic Regression |
| **Web UI** | `streamlit` | Interactive frontend dashboard |
| **Visualization** | `matplotlib`, `seaborn` | Analytics charts and heatmaps |
| **Dataset Format** | CSV | Structured tabular data storage |

---

## Machine Learning Pipeline

```
Raw Text Fields
(professional_summary + about_me)
         │
         ▼
┌─────────────────────┐
│  Text Preprocessing │
│  • Lowercasing      │
│  • Punctuation strip│
│  • Stopword removal │
│  • Lemmatization    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF Vectorizer  │
│  • max_features=5000│
│  • ngram_range=(1,2)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│       Similarity Matrix             │
│  cosine_similarity(tfidf_matrix)    │
└────────┬────────────────────────────┘
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌──────────────────┐            ┌───────────────────────┐
│  Content Score   │            │   Feedback Model      │
│  (NLP Cosine)    │            │   Logistic Regression │
│  Weight: 0.70    │            │   Weight: 0.30        │
└────────┬─────────┘            └──────────┬────────────┘
         │                                 │
         └──────────────┬──────────────────┘
                        ▼
             ┌─────────────────────┐
             │   Hybrid Score      │
             │   Final Ranked List │
             └─────────────────────┘
```

**Feedback Labels:**
- `connect` → Positive (1)
- `like` → Positive (1)
- `skip` → Negative (0)

---

## Dataset

The project uses a custom dataset hosted on Kaggle.

🔗 **[ProfileMatch Dataset on Kaggle](https://www.kaggle.com/datasets/debabratakuiry/profile-matching-and-recommendation-dataset)**

### `users.csv`

| Field | Type | Description |
|---|---|---|
| `user_id` | int | Unique user identifier |
| `name` | string | Full name |
| `age` | int | User age |
| `location` | string | City / Country |
| `profession` | string | Job title or field |
| `experience_years` | int | Years of professional experience |
| `professional_summary` | text | Career background paragraph |
| `about_me` | text | Personal interests and personality |
| `mbti` | string | Myers-Briggs personality type |
| `interests` | string | Comma-separated interests |

### `feedback.csv`

| Field | Type | Description |
|---|---|---|
| `user_id` | int | Initiating user |
| `matched_user_id` | int | Profile that was reviewed |
| `action` | string | `like` / `skip` / `connect` |
| `timestamp` | datetime | When the action was taken |

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Recommended) Virtual environment

### Step-by-Step Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/ProfileMatch.git
cd ProfileMatch
```

**2. Create and activate a virtual environment**
```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download NLTK resources**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

**5. Add the dataset**
```
Place users.csv and feedback.csv inside the data/ directory.
```

**6. Launch the application**
```bash
streamlit run app/app.py
```

---

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## Project Structure

```
ProfileMatch/
│
├──  data/
│   ├── users.csv                     # User profile dataset
│   └── feedback.csv                  # User interaction feedback
│
├──  src/
│   ├── preprocessing.py              # Data cleaning & NLP preprocessing
│   ├── learning.py                   # Generates feature vectors using TF-IDF vectorization
│   ├── similarity.py                 # Computes similarity scores between user profiles using cosine similarity
│   └── scoring.py                    # Calculates compatibility scores and updates the recommendation model
│
├── app/                        
│   ├── app.py                        # Streamlit web application 
├── requirements.txt                  # Python dependencies
├── LICENSE
└── README.md
```

---

##  Performance Metrics

| Metric | Score | Notes |
|---|---|---|
| **TF-IDF Cosine Similarity (Avg)** | 0.847 | Evaluated on 500 profile pairs |
| **Feedback Model Accuracy** | 88.3% | Logistic Regression on train/test split |
| **Precision@5** | 0.821 | Top-5 recommendations |
| **Recall@10** | 0.793 | Top-10 recommendations |
| **F1-Score (Feedback)** | 0.856 | Binary like/skip classification |
| **Mean Reciprocal Rank (MRR)** | 0.874 | Ranking quality metric |
| **Average Response Time** | < 300ms | Streamlit UI on local machine |

>  Metrics computed using 80/20 train-test split with stratified sampling on `feedback.csv`.

---

## Use Cases

-  **Professional Networking** — Match professionals based on skills, experience, and career goals
-  **Academic Collaboration** — Connect researchers and students with aligned interests
-  **Talent Acquisition** — Help recruiters shortlist candidates that match team culture and role
-  **Social Discovery** — Surface compatible users in community platforms
-  **Enterprise Team Building** — Recommend cross-functional partners with complementary profiles
-  **Mentorship Platforms** — Pair mentors and mentees based on goals and expertise overlap

---

## Future Enhancements

- [ ]  **Deep Learning Matching** — Replace TF-IDF with sentence-transformers (BERT/SBERT) for richer semantic understanding
- [ ]  **Multilingual Support** — Extend NLP pipeline to handle profiles in multiple languages
- [ ]  **Graph-Based Recommendations** — Model user interactions as a graph using GNNs for collaborative filtering
- [ ]  **Authentication & Profiles** — Add user login, profile creation, and session management
- [ ]  **REST API** — Expose matching engine as a FastAPI microservice for third-party integration
- [ ]  **Docker Support** — Containerize the application for one-command deployment
- [ ]  **Cloud Deployment** — Deploy to Streamlit Cloud / AWS / GCP with CI/CD pipeline
- [ ]  **Real-Time Notifications** — Alert users of new high-compatibility matches

---

## Contributing

Contributions are warmly welcomed! Here's how to get started:

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/ProfileMatch.git

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Commit your changes
git commit -m "feat: add your feature description"

# 5. Push to your branch
git push origin feature/your-feature-name

# 6. Open a Pull Request on GitHub
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 ProfileMatch Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## Author

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/50688c5b-e7cb-48a7-b7dc-725351480a58" width="100px;" alt="Author"/><br/>
      <a href="https://linkedin.com/in/debabrata_kuiry/">LinkedIn</a> •
      <a href="mailto:debabratakuiry2002@gmail.com">Email</a>
    </td>
  </tr>
</table>

---

<p align="center">
  <sub> If you found this project helpful, please consider giving it a star — it helps others discover it!</sub><br/><br/>
  <img src="https://img.shields.io/github/stars/cyrax3589/ProfileMatch?style=social" />
  <img src="https://img.shields.io/github/forks/cyrax3589/ProfileMatch?style=social" />
</p>

<p align="center">
  Intelligent matching powered by machine learning · <a href="#-profilematch">Back to Top ↑</a>
</p>
