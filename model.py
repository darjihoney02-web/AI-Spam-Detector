"""
SpamShield AI — model.py
Handles training, prediction, explanation, and advanced signal detection.
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import joblib
import os

# ─────────────────────────────────────────────
# SIGNAL DICTIONARIES
# ─────────────────────────────────────────────

URGENCY_WORDS = [
    "urgent", "immediately", "hurry", "limited time", "act now", "don't wait",
    "expires", "deadline", "last chance", "right now", "asap", "today only",
    "final notice", "respond now", "alert", "warning", "attention"
]

PHISHING_WORDS = [
    "verify", "account", "password", "login", "bank", "click here",
    "confirm", "update", "suspended", "locked", "credential", "security",
    "unusual activity", "validate", "access", "reset", "blocked"
]

PROMO_WORDS = [
    "free", "win", "winner", "prize", "cash", "reward", "offer",
    "discount", "deal", "sale", "bonus", "gift", "claim", "congratulations",
    "selected", "lucky", "exclusive", "subscribe", "trial"
]

# ─────────────────────────────────────────────
# TRAIN MODEL
# ─────────────────────────────────────────────

def train_model():
    """Train a TF-IDF + Naive Bayes pipeline on the spam dataset."""
    csv_path = "dataset.csv"
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "dataset.csv")

    data = pd.read_csv(csv_path)
    data.dropna(subset=["label", "message"], inplace=True)
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    data.dropna(subset=["label"], inplace=True)

    X = data["message"].astype(str)
    y = data["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    )

    model = MultinomialNB(alpha=0.1)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, round(acc * 100, 2)

# ─────────────────────────────────────────────
# PREDICTION & EXPLANATION
# ─────────────────────────────────────────────

def predict_message(text: str, model, vectorizer):
    """
    Returns a rich dict with:
      - prediction, spam_prob, ham_prob
      - top_words (list of (word, score))
      - urgency_signals, phishing_signals, promo_signals
      - risk_level, explanation
    """
    vec = vectorizer.transform([text])
    prediction = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]
    spam_prob = round(float(proba[1]) * 100, 2)
    ham_prob  = round(float(proba[0]) * 100, 2)

    # --- Top contributing TF-IDF words ---
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores  = vec.toarray()[0]
    word_indices  = np.argsort(tfidf_scores)[::-1]
    top_words = [
        (feature_names[i], round(float(tfidf_scores[i]), 4))
        for i in word_indices
        if tfidf_scores[i] > 0
    ][:10]

    # --- Signal detection ---
    lower = text.lower()
    urgency_hits  = [w for w in URGENCY_WORDS   if w in lower]
    phishing_hits = [w for w in PHISHING_WORDS  if w in lower]
    promo_hits    = [w for w in PROMO_WORDS      if w in lower]

    # --- Risk level ---
    if spam_prob >= 85:
        risk_level = "Critical"
    elif spam_prob >= 60:
        risk_level = "High"
    elif spam_prob >= 35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # --- Plain-language explanation ---
    reasons = []
    if top_words:
        reasons.append(f"Key trigger words: {', '.join(w for w, _ in top_words[:5])}")
    if urgency_hits:
        reasons.append(f"Urgency signals detected: {', '.join(urgency_hits[:3])}")
    if phishing_hits:
        reasons.append(f"Phishing indicators found: {', '.join(phishing_hits[:3])}")
    if promo_hits:
        reasons.append(f"Promotional language: {', '.join(promo_hits[:3])}")
    if not reasons:
        reasons.append("No strong spam signals detected in this message.")

    # --- Spam sub-type ---
    if phishing_hits:
        spam_type = "Phishing"
    elif promo_hits:
        spam_type = "Promotional Spam"
    elif urgency_hits:
        spam_type = "Urgency Manipulation"
    else:
        spam_type = "General Spam" if prediction == 1 else "Legitimate"

    return {
        "prediction":      prediction,
        "spam_prob":       spam_prob,
        "ham_prob":        ham_prob,
        "risk_level":      risk_level,
        "spam_type":       spam_type,
        "top_words":       top_words,
        "urgency_signals": urgency_hits,
        "phishing_signals":phishing_hits,
        "promo_signals":   promo_hits,
        "explanation":     " | ".join(reasons),
    }

# ─────────────────────────────────────────────
# BULK ANALYSIS
# ─────────────────────────────────────────────

def analyze_bulk(messages: list[str], model, vectorizer) -> list[dict]:
    """Analyze a list of messages and return results for each."""
    return [predict_message(msg, model, vectorizer) for msg in messages if msg.strip()]