# Satyam AI — Fake News Detection System

Welcome to Satyam AI, a smart, AI-powered fake news detection system built during **Codewizards'25** at **SRMIST Delhi NCR Campus**. This project was designed to combat the growing problem of misinformation by using deep learning and NLP techniques to distinguish between real and fake news articles in real time.

> "Satyam" means truth — and this project is built to help you find it.

---

## Live Status

- Built and demoed at **Codewizards'25**
- Improved post-hackathon with smarter logic and a better interface
- Clean desktop interface using Tkinter

---

## Features

- **Real vs. Fake Prediction:** Instantly classifies a news headline or article as genuine or fake.
- **Deep Learning Engine:** Uses LSTM + Word2Vec for contextual text analysis.
- **Simple UI:** A clean, local interface for user input and prediction.
- **Improved Accuracy:** Post-hackathon logic refinements for better results.
- **Explainable Outputs:** Potential to expand with attention visualization or keyword highlighting.

---

## Tech Stack

| Component         | Technology                  |
|:------------------|:----------------------------|
| **Language**      | Python 3                    |
| **Frontend**      | Tkinter                     |
| **NLP Embeddings**| Word2Vec (Gensim)           |
| **Deep Learning** | LSTM (Keras/TensorFlow)     |
| **Data Handling** | Pandas, NumPy               |
| **Visualization** | Matplotlib, Seaborn (optional)|

---

## Getting Started

1.  **Clone the Repo**
    ```bash
    git clone https://github.com/tarunbarkoti/satyam-ai.git
    cd satyam-ai
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    python app.py
    ```
    > **Note:** If using the Tkinter GUI, run `gui.py` instead.

---

## Model Overview

- Trained on a curated dataset of news articles labeled as real or fake.
- Word2Vec used to convert text to embeddings.
- LSTM model learns sequence-based context.
- Enhanced with better preprocessing (like stopwords removal, stemming, etc.).

---

## Developed By

**Tarun Barkoti**

- [LinkedIn](https://www.linkedin.com/in/tarunbarkoti/)

---

## License

This project is licensed under the **MIT License**.
