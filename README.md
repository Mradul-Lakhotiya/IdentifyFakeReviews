# IdentifyFakeReviews

Welcome to **IdentifyFakeReviews** — a personal learning project about classifying text reviews and identifying fake reviews using various machine learning and deep learning approaches.

Maintained by [Mradul-Lakhotiya](https://github.com/Mradul-Lakhotiya).

---

## 🧑‍💻 About This Project

This repository is my attempt at building and iterating on a text classification pipeline for detecting fake reviews.  
You’ll find multiple versions and approaches here: from classical ML pipelines (TF-IDF, Word2Vec, feature engineering, scikit-learn models) to transformer-based architectures (BERT).

> **Note:** This project is not production-ready. It was a learning experience for me.  
> I experimented with several techniques and learned important lessons:
> 
> - How to use transformers and encoders (BERT, Word2Vec, etc.)
> - Why simple MLPs don’t work well on raw text
> - The importance of transfer learning for NLP
> - Overfitting challenges in text data
> 
> There are flaws: some models overfit, and the code is experimental. But the journey taught me a lot about text classification and modern NLP.

---

## 🗂️ Code Organization

- **app.py:** Flask app serving BERT-based predictions (`nlp_model.pth` required).
- **Version1.0, Version2.0, Version3.0:**  
  Multiple iterations, each with different pipelines, notebooks, and utility scripts:
    - **ml_utils.py:** Custom scikit-learn transformers for feature selection, missing value imputation, and more.
    - **DataPipeline.py:** Text preprocessing, lemmatization, tokenization, Word2Vec training.
    - **main.ipynb, data.ipynb:** Notebooks for feature engineering, model training, and EDA.
- [View all code files and versions here.](https://github.com/Mradul-Lakhotiya/IdentifyFakeReviews)

---

## ✨ Features & Techniques

- Text cleaning, lemmatization, and tokenization
- Feature extraction (TF-IDF, Word2Vec, BERT embeddings)
- Classical ML models: Logistic Regression, SVM, Random Forest, Naive Bayes
- Deep Learning: BERT for sequence classification
- Model evaluation and deployment (Flask API)

---

## 🚀 How to Use

1. **Clone the repository.**
2. **Explore different versions** (`Version1.0`, `Version2.0`, `Version3.0`) for code evolution.
3. **Train models** using provided notebooks/scripts.
4. **Run predictions** with Flask app (requires trained BERT model weights).
5. **Experiment, learn, and adapt!**

---

## 🤝 Contributing

Contributions are welcome, but please note this is primarily a learning and experimental repo.
If you wish to extend or refactor, feel free to fork and open a pull request!

---

_For questions, suggestions, or to collaborate, open an issue or pull request!_

---