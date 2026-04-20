<div align="center">

<!-- HERO BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,20,24&height=200&section=header&text=🛡️%20SpamShield%20AI&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Intelligent%20Spam%20Detection%20%7C%20Powered%20by%20NLP%20%26%20Machine%20Learning&descAlignY=58&descSize=16&animation=fadeIn" width="100%" alt="SpamShield AI Banner"/>

<br/>

<!-- BADGES -->
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TF-IDF](https://img.shields.io/badge/NLP-TF--IDF-00D4FF?style=for-the-badge&logo=databricks&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![Hackathon](https://img.shields.io/badge/Hackathon-Ready-7c3aed?style=for-the-badge)

<br/><br/>

> **"Stop spam before it reaches you."**
>
> SpamShield AI is a real-time, explainable spam detection platform that goes beyond simple filtering —
> it tells you *why* a message is spam, *what type* of threat it is, and *how confident* the AI is.

<br/>

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Try%20It%20Now-00D4FF?style=for-the-badge)]([https://your-demo-link.streamlit.app](https://ai-spam-detector-pwcgtj6xuyded9guup4m6d.streamlit.app/))
[![View Source](https://img.shields.io/badge/📂%20Source%20Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/yourusername/spamshield-ai)
[![Report Bug](https://img.shields.io/badge/🐛%20Report%20Bug-Issues-FF4B4B?style=for-the-badge)](https://github.com/yourusername/spamshield-ai/issues)

</div>

---

<!-- DEMO GIF -->
<div align="center">

## 🎬 See It In Action

<img src="https://via.placeholder.com/900x500/0d1117/00d4ff?text=📸+App+Demo+GIF+—+Replace+with+your+screen+recording" width="90%" alt="SpamShield AI Demo"/>

<sub>💡 <i>Replace the above with a screen recording GIF — tools like <a href="https://www.screentogif.com">ScreenToGif</a> or <a href="https://obsproject.com">OBS</a> work great.</i></sub>

</div>

---

## 📌 Table of Contents

| # | Section |
|---|---------|
| 1 | [🚀 Introduction](#-introduction) |
| 2 | [🎯 Problem Statement](#-problem-statement) |
| 3 | [💡 Our Solution](#-our-solution) |
| 4 | [✨ Features](#-features) |
| 5 | [🧠 AI & ML Architecture](#-ai--ml-architecture) |
| 6 | [🖥️ UI Highlights](#️-ui-highlights) |
| 7 | [📂 Project Structure](#-project-structure) |
| 8 | [⚡ Installation & Usage](#-installation--usage) |
| 9 | [🌍 Real-World Applications](#-real-world-applications) |
| 10 | [🔭 Future Scope](#-future-scope) |
| 11 | [🏆 Why SpamShield Stands Out](#-why-spamshield-stands-out) |
| 12 | [🤝 Contributing](#-contributing) |
| 13 | [📜 License](#-license) |

---

## 🚀 Introduction

Every day, **over 3.4 billion spam emails** are sent worldwide. Spam isn't just annoying — it's dangerous. Phishing attacks, fraudulent messages, and urgency manipulation cost individuals and businesses **billions of dollars annually**.

**SpamShield AI** is a production-grade, AI-powered spam detection platform built to solve this problem with transparency, speed, and intelligence. Unlike traditional spam filters that silently block messages, SpamShield **explains every decision** — giving users full visibility into how and why a message was flagged.

Built with a modern SaaS-grade interface, SpamShield is designed to impress in demos and deliver real value in production.

---

## 🎯 Problem Statement

<table>
<tr>
<td width="50%">

### ❌ The Problem

- Spam filters are **black boxes** — users never know why a message was blocked
- Basic keyword filters **miss sophisticated spam** like phishing and urgency manipulation
- No existing lightweight tool provides **explainable AI** for spam detection
- Most tools are enterprise-only — nothing accessible for students, developers, or small teams
- Bulk analysis of message datasets requires **custom scripting** for most users

</td>
<td width="50%">

### ✅ What We Needed

- A **transparent** system that explains each prediction in plain language
- Detection of **multiple spam types** (phishing, promotional, urgency)
- A **modern UI** that non-technical users can operate confidently
- A **fast, lightweight** solution that runs locally or on the cloud
- **Bulk analysis** capability without writing any code

</td>
</tr>
</table>

---

## 💡 Our Solution

SpamShield AI combines classical machine learning with modern UI/UX design to deliver a tool that is simultaneously powerful, explainable, and beautiful.

```
User Input (text / CSV file)
          │
          ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌───────────────────────┐
│   TF-IDF            │───▶│   Naive Bayes         │───▶│   Explainability      │
│   Vectorizer        │    │   Classifier          │    │   Engine              │
│   (NLP Layer)       │    │   (Prediction Layer)  │    │   (Signal Detection)  │
└─────────────────────┘    └──────────────────────┘    └───────────────────────┘
                                                                  │
                            ┌─────────────────────────────────────┘
                            ▼
                 ┌──────────────────────────────────────────┐
                 │          Streamlit Dashboard              │
                 │  Chat UI · Charts · Risk Cards · Bulk    │
                 └──────────────────────────────────────────┘
```

**Three layers of intelligence:**

1. **NLP Layer** — TF-IDF with bigrams extracts meaningful patterns from raw text
2. **Prediction Layer** — Multinomial Naive Bayes classifies with calibrated probabilities
3. **Explainability Layer** — Signals engine surfaces urgency, phishing, and promotional patterns

---

## ✨ Features

### 🔍 Core Detection

| Feature | Description | Status |
|---------|-------------|--------|
| **Real-time Spam Detection** | Classify any message instantly as spam or safe | ✅ Live |
| **Spam Probability Score** | Exact confidence percentage, not just a binary label | ✅ Live |
| **Risk Level Classification** | Critical / High / Medium / Low risk tiers | ✅ Live |
| **Spam Sub-type Detection** | Phishing · Urgency Manipulation · Promotional · General | ✅ Live |
| **Bulk CSV Analysis** | Upload a file, analyse all rows, download results | ✅ Live |

### 🧠 Explainable AI

| Feature | Description | Status |
|---------|-------------|--------|
| **Trigger Word Highlighting** | Risky words highlighted directly inside the message text | ✅ Live |
| **TF-IDF Word Scores** | Top contributing words with their importance scores shown | ✅ Live |
| **Urgency Signal Detection** | Flags "urgent", "act now", "deadline", "expires" | ✅ Live |
| **Phishing Signal Detection** | Flags credential-harvesting and account-threat language | ✅ Live |
| **Plain-language Explanation** | Human-readable verdict reason, not just a number | ✅ Live |

### 🎨 UI / UX

| Feature | Description | Status |
|---------|-------------|--------|
| **Chat-style Interface** | ChatGPT-style conversation UI for interactive analysis | ✅ Live |
| **Dark / Light Mode** | One-click theme toggle with full re-theme | ✅ Live |
| **Animated Confidence Bars** | Visual spam vs safe probability bars | ✅ Live |
| **Interactive Donut Chart** | Plotly probability ring for each prediction | ✅ Live |
| **Glassmorphism Design** | Modern SaaS-grade frosted glass UI cards | ✅ Live |

### 📊 Analytics Dashboard

| Feature | Description | Status |
|---------|-------------|--------|
| **Session Dashboard** | Full analytics of all messages analysed this session | ✅ Live |
| **Spam vs Safe Pie Chart** | Visual session-level breakdown | ✅ Live |
| **Spam Probability Timeline** | Line chart of risk scores across messages | ✅ Live |
| **Top Trigger Words Chart** | Bar chart of the most frequent risk keywords | ✅ Live |
| **Bulk Summary Report** | Type breakdown and stats for uploaded CSV files | ✅ Live |

---

## 🧠 AI & ML Architecture

### Model Pipeline

```python
# Step 1 — Feature Extraction (TF-IDF with bigrams)
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),      # captures phrases like "click here", "act now"
    stop_words="english",
    sublinear_tf=True         # reduces impact of very frequent terms
)

# Step 2 — Classification (Multinomial Naive Bayes)
MultinomialNB(alpha=0.1)     # Laplace smoothing for unseen vocabulary

# Step 3 — Output (Probability + Label)
model.predict_proba(X)       # returns [ham_probability, spam_probability]
```

### Why TF-IDF over CountVectorizer?

| Metric | CountVectorizer | TF-IDF (SpamShield) |
|--------|----------------|----------------------|
| Handles common words | ❌ Treats all equally | ✅ Penalises frequent words |
| Bigram support | ⚠️ Optional | ✅ Enabled by default |
| Sublinear term scaling | ❌ Not available | ✅ Applied |
| Real-world performance | Baseline | **Meaningfully higher** |

### Signal Detection Dictionary

```
🎣 PHISHING    →  verify, account, password, login, suspended, credential, validate
⏰ URGENCY     →  urgent, act now, limited time, expires, deadline, asap, final notice
📢 PROMOTIONAL →  free, win, prize, cash, reward, claim, congratulations, selected
🚫 GENERAL     →  catch-all for patterns not matching the above three categories
```

---

## 🖥️ UI Highlights

<div align="center">

| Chat Interface | Analytics Dashboard | Bulk Analysis |
|:---:|:---:|:---:|
| <img src="https://via.placeholder.com/260x180/0d1117/00d4ff?text=Chat+UI" width="260" alt="Chat UI"/> | <img src="https://via.placeholder.com/260x180/0d1117/7c3aed?text=Analytics" width="260" alt="Analytics"/> | <img src="https://via.placeholder.com/260x180/0d1117/22c55e?text=Bulk+Upload" width="260" alt="Bulk"/> |
| ChatGPT-style bubbles with risk badges | Live charts: pie, line, bar, word frequency | Upload CSV → Analyse all → Download results |

</div>

> 💡 Replace the placeholder images above with actual screenshots from your running app.

---

## 📂 Project Structure

```
SpamShield-AI/
│
├── 📄 app.py                ← Main Streamlit application (UI layer only)
├── 🧠 model.py              ← ML model: training, prediction, explainability
├── 🛠️  utils.py             ← Helpers: highlighting, labels, risk formatting
├── 📊 dataset.csv           ← Training data (label + message columns)
├── 📋 requirements.txt      ← Python dependencies
└── 📖 README.md             ← You are here
```

**Architecture principle:** Strict separation of concerns.

- `app.py` handles only UI — it never performs ML logic directly
- `model.py` handles only ML — it has zero Streamlit imports
- `utils.py` handles only formatting — pure functions, no side effects

---

## ⚡ Installation & Usage

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/spamshield-ai.git
cd spamshield-ai
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

Packages installed:

```
pandas          → Data loading and manipulation
scikit-learn    → TF-IDF vectorizer + Naive Bayes classifier
streamlit       → Web application framework
plotly          → Interactive charts and visualisations
```

### Step 3 — Run the App

```bash
streamlit run app.py
```

Open your browser at: **`http://localhost:8501`**

### Step 4 — Using the App

```
1. Type any message into the input field
2. Click "Analyse Message"
3. View verdict, confidence score, risk level, and explanation
4. Switch to "Bulk Upload" to analyse an entire CSV file
5. Open "Analytics Dashboard" to view session-wide stats
```

### Dataset Format

Your `dataset.csv` must follow this exact structure:

```csv
label,message
ham,Hey, are you free tomorrow?
spam,URGENT! You have won a FREE prize — click here to claim NOW!
ham,Please find the report attached to this email.
spam,Verify your bank account immediately or it will be suspended.
```

### Deployment Options

| Platform | Difficulty | Cost | Command |
|----------|------------|------|---------|
| **Streamlit Cloud** | ⭐ Easy | Free | Push to GitHub → connect at share.streamlit.io |
| **Railway** | ⭐⭐ Medium | Free tier | Add `Procfile` → connect repo |
| **Docker** | ⭐⭐ Medium | Self-hosted | `docker build -t spamshield . && docker run -p 8501:8501 spamshield` |
| **Render** | ⭐⭐ Medium | Free tier | Add start command in dashboard |

**Procfile for Railway / Render:**
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 🌍 Real-World Applications

<table>
<tr>
<td align="center" width="25%"><br/>📧<br/><b>Email Providers</b><br/><sub>Filter inboxes at scale with explainable flagging</sub></td>
<td align="center" width="25%"><br/>📱<br/><b>SMS Platforms</b><br/><sub>Protect users from smishing and urgency scams</sub></td>
<td align="center" width="25%"><br/>🏢<br/><b>Enterprise IT</b><br/><sub>Bulk-screen internal communication channels</sub></td>
<td align="center" width="25%"><br/>🎓<br/><b>Education</b><br/><sub>Teach NLP and explainable AI with a live demo</sub></td>
</tr>
<tr>
<td align="center" width="25%"><br/>🛍️<br/><b>E-commerce</b><br/><sub>Screen product reviews and buyer messages</sub></td>
<td align="center" width="25%"><br/>🏦<br/><b>FinTech</b><br/><sub>Flag phishing targeting account holders</sub></td>
<td align="center" width="25%"><br/>🔬<br/><b>Research</b><br/><sub>Baseline model for NLP classification studies</sub></td>
<td align="center" width="25%"><br/>🌐<br/><b>Social Media</b><br/><sub>Moderate user-generated content at volume</sub></td>
</tr>
</table>

---

## 🔭 Future Scope

| Priority | Feature | Description |
|----------|---------|-------------|
| 🔴 High | **Transformer Model** | Upgrade to BERT / DistilBERT for 95%+ accuracy |
| 🔴 High | **Multilingual Detection** | Hindi, Spanish, French, and Arabic spam support |
| 🟡 Medium | **Voice Input** | Speech-to-text for spoken message analysis |
| 🟡 Medium | **Browser Extension** | Real-time detection in Gmail and Outlook |
| 🟡 Medium | **REST API** | FastAPI backend so any app can query SpamShield |
| 🟢 Low | **Feedback Loop** | Users correct predictions to continuously retrain |
| 🟢 Low | **Threat Intelligence** | Cross-reference against known phishing URL databases |
| 🟢 Low | **Mobile App** | React Native wrapper for iOS and Android |

---

## 🏆 Why SpamShield Stands Out

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║              What Makes SpamShield Different                 ║
╠══════════════════════════════════════════════════════════════╣
║  ✅  Explainable AI — not just a label, but a reason         ║
║  ✅  4 spam sub-types — phishing, urgency, promo, general    ║
║  ✅  Word-level highlighting — see exactly what triggered it  ║
║  ✅  Bulk analysis — CSV in, results CSV out                  ║
║  ✅  Session analytics dashboard — charts, trends, history    ║
║  ✅  Production UI — glassmorphism, dark/light, animations    ║
║  ✅  Clean modular code — 3 files, strict separation          ║
║  ✅  Deployable in 2 commands — zero configuration needed     ║
╚══════════════════════════════════════════════════════════════╝
```

</div>

### 🎯 Hackathon Impact

- **Problem scope:** Billions of spam messages sent daily — massive, real, and global
- **Technical depth:** Full NLP pipeline + explainability engine + signal detection layer
- **Demo-ability:** Live app, interactive UI, instant results — judges can try it themselves in seconds
- **Code quality:** Modular, documented, beginner-readable but professionally structured
- **Extensibility:** Clear upgrade path to transformers, REST APIs, and mobile platforms

---

## 🤝 Contributing

Contributions are welcome and appreciated. Here is how to get involved:

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/yourusername/spamshield-ai.git

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes and commit
git add .
git commit -m "feat: describe what you added"

# 5. Push and open a Pull Request
git push origin feature/your-feature-name
```

**Good first contribution ideas:**

- 🌐 Add multilingual spam dataset support
- 🤖 Integrate a transformer-based model option
- 🧪 Write unit tests for `model.py` and `utils.py`
- 📱 Build a REST API wrapper using FastAPI
- 🎨 Improve animations and mobile responsiveness
- 📊 Add more analytics chart types to the dashboard

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

```
MIT License — Copyright (c) 2025 SpamShield AI Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to deal in the Software without restriction, including the
rights to use, copy, modify, merge, publish, distribute, and sublicense.
```

See the full [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Scikit-learn](https://scikit-learn.org) — ML pipeline and TF-IDF vectorizer
- [Streamlit](https://streamlit.io) — Python web app framework
- [Plotly](https://plotly.com) — Interactive data visualisations
- [Shields.io](https://shields.io) — README badge generation
- [Capsule Render](https://github.com/kyechan99/capsule-render) — Hero banner images

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,20,24&height=120&section=footer&animation=fadeIn" width="100%" alt="Footer Wave"/>

**Built with ❤️ by the SpamShield AI team**

*If this project helped you or impressed you, please consider giving it a ⭐*
*It helps others discover the project and motivates continued development.*

[![GitHub Stars](https://img.shields.io/github/stars/yourusername/spamshield-ai?style=social)](https://github.com/yourusername/spamshield-ai)
[![GitHub Forks](https://img.shields.io/github/forks/yourusername/spamshield-ai?style=social)](https://github.com/yourusername/spamshield-ai/fork)
[![Follow](https://img.shields.io/github/followers/yourusername?style=social)](https://github.com/yourusername)

<br/>

`Python` · `Streamlit` · `TF-IDF` · `Naive Bayes` · `Explainable AI` · `NLP` · `Open Source`

</div>
