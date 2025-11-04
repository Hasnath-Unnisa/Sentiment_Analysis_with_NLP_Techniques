# 🎯 Sentiment & Emotion Analysis Using NLP Techniques

This project applies **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to classify text data
into emotions such as *happy* or *frustrated*.  
The workflow follows the **CRISP–DM methodology**, demonstrating each phase — from business understanding to evaluation.

---

## 📘 Project Overview

This notebook demonstrates an **end-to-end sentiment analysis pipeline**:
- Text preprocessing (tokenization, stopword removal, lemmatization)
- Feature extraction using **TF-IDF vectorization**
- Training and comparison of multiple ML models
- Evaluation and insights on model behavior and data quality

---

## 🧩 Objectives
- Understand and preprocess raw text data  
- Apply vectorization to convert text into numeric form  
- Train and evaluate several classification models  
- Analyze performance limitations on small/imbalanced datasets  

---

## 📊 Dataset
- **Total samples:** 25 (≈30 KB)
- **Classes:** `happy`, `frustrated`
- **Features after TF-IDF:** 1 404
- **Train/Test split:** 80 % / 20 %

> ⚠️ *This dataset is for demonstration only — it is too small for production-level modeling.*

---

## ⚙️ Methodology

| Phase | Description |
|:--|:--|
| **Business Understanding** | Define the goal — classify emotions in short text reviews. |
| **Data Understanding** | Explore the dataset, check class balance and sample distribution. |
| **Data Preparation** | Clean text, remove stopwords, tokenize & lemmatize, create TF-IDF features. |
| **Modeling** | Train models — Naive Bayes, Logistic Regression, KNN, Decision Tree, Random Forest. |
| **Evaluation** | Compare results using Accuracy, Precision, Recall, and F1-Score. |
| **Deployment** | Demonstrate prediction logic; ready for integration via Flask/FastAPI. |

---

## 🧮 Model Evaluation Results

| Metric | Training | Testing |
|:--|:--|:--|
| Accuracy | 0.80 | 0.40 |
| Precision | 0.84 | 0.16 |
| Recall | 0.80 | 0.40 |
| F1-Score | 0.80 | 0.23 |

**Confusion Matrix**

|            | Predicted Frustrated | Predicted Happy |
|-------------|----------------------|-----------------|
| **Actual Frustrated** | 0 | 3 |
| **Actual Happy** | 0 | 2 |

> The model tends to overpredict *happy* and fails to capture *frustrated* due to the small, imbalanced dataset.  
> Overfitting is visible: excellent training accuracy, poor test performance.

---

## 🧠 Insights & Observations
- **KNN**, **Decision Tree**, and **Random Forest** overfit heavily on the small dataset.  
- **Logistic Regression** provides a consistent baseline but still limited by data size.  
- **Hyperparameter tuning** did not improve results — lack of data is the main bottleneck.  
- The pipeline itself is **reproducible and educational** for NLP beginners.

---

## 🚀 Future Improvements
- Collect a **larger balanced dataset** (≥ 1 000 samples)
- Apply **SMOTE** or other resampling for class balance
- Use **cross-validation**
- Experiment with **Word2Vec**, **LSTM**, or **BERT** embeddings
- Deploy model via **Flask** or **FastAPI** for live sentiment prediction

---

## 🧰 Tech Stack
- **Language:** Python 3  
- **Libraries:**  
  `pandas`, `numpy`, `nltk`, `scikit-learn`, `spacy`, `matplotlib`, `seaborn`, `wordcloud`, `PyPDF2`, `python-docx`

---

## 🧾 Project Structure

```
NLP_Sentiment_Analysis/
│
├── Data/ # Dataset and cleaned data
├── Source_code/ # Jupyter notebook (Sentiment_Analysis.ipynb)
├── Documentation/ #Report
├── requirements.txt # Python dependencies
├── .gitignore # Ignore unnecessary files
└── README.md # 
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Hasnath-Unnisa/Sentiment_Analysis_with_NLP_Techniques.git
   cd Sentiment_Analysis_with_NLP_Techniques

2. Create virtual environment

python -m venv .venv
source .venv/Scripts/activate   # Windows

3.Install dependencies

pip install -r requirements.txt

4. Run the notebook

jupyter notebook Source_code/Sentiment_Analysis.ipynb

🏁 Results Summary

Best Model: Logistic Regression

Accuracy: 40 %

Observation: Overfitting due to small dataset

Recommendation: Increase dataset size and test deep learning models

## 👩‍💻 Author  

**Name:** Hasnath Unnisa  
**Email:** unnisahasnath@gmail.com  
**LinkedIn:** www.linkedin.com/in/hasnath22  
