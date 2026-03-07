# NLP Disaster Tweet Classifier

## Overview

Twitter is one of the fastest ways information spreads during emergencies — but not every alarming tweet is about a real disaster. Phrases like "the traffic was on fire" or "this weather is destroying me" use disaster language without describing actual events. This project builds a classifier that distinguishes real disaster tweets from figurative ones, using a dataset of 7,613 labeled tweets from a Kaggle NLP competition.

The best model, a Logistic Regression trained on bag-of-words features, achieved **80% accuracy** — outperforming an LSTM Neural Network and a Random Forest on this task. This project was completed as a final for a graduate course ECON 524 Advanced Machine Learning at Washington State University.

---

## The Problem

Emergency response systems, news aggregators, and crisis monitoring tools could benefit from automated tweet filtering. The challenge is that disaster-related words appear constantly in non-disaster contexts. A reliable classifier needs to go beyond keyword detection and learn contextual patterns — tone, structure, and phrasing — that separate a news report from a casual post.

---

## Dataset

The data comes from Kaggle's *Natural Language Processing with Disaster Tweets* competition. Each tweet is labeled:
- **1** — the tweet describes a real disaster
- **0** — the tweet does not

The dataset contains 7,613 tweets, with 43% labeled as disaster tweets and 57% as non-disasters. Each row also includes a keyword (the alarming word that triggered inclusion) and an optional location field.

---

## Feature Engineering

A significant part of this project was expanding the feature space to capture as much signal as possible without overfitting. Raw tweet text alone is noisy — it contains URLs, @mentions, stopwords, and inconsistent formatting. The following features were constructed:

**Text Cleaning**
- Removed URLs, Twitter @mentions, stopwords, and punctuation
- Lowercased all text for consistency

**Keyword Dummies**
- 222 unique keywords were each converted into a binary dummy variable, creating 222 additional features. Keywords like "wildfire" or "earthquake" carry strong predictive signal.

**Formality Score**
- Disaster tweets tended to be longer (avg. 77 characters) than non-disaster tweets (avg. 68 characters), consistent with the observation that news station tweets — which skew formal — were almost always real disasters. A binary formality dummy was created using a threshold of 67 characters.

**News Dummy**
- Tweets containing the word "news" and labeled as a disaster were flagged, capturing the pattern that news outlet tweets were reliably real events.

**Symbol Count**
- Disaster tweets contained far fewer exclamation marks and question marks (avg. 0.33) than non-disaster tweets (avg. 0.74), reflecting the difference between formal reporting and casual expression.

**Geographic Features**
- SpaCy's NER model was used to extract city names from the messy location column, which was then one-hot encoded.

**Bag-of-Words (CountVectorizer)**
- The cleaned tweet text was tokenized and vectorized, expanding the feature space to approximately 18,000 word-level features for use in the Logistic Regression and Random Forest models.

---

## Models

Three models were trained and evaluated:

**LSTM Neural Network (no scaling)**
- Tokenized sequences padded to length 100, embedded into 100-dimensional space
- SpatialDropout and LSTM layer with 50% dropout
- Test Accuracy: **76.8%**

**LSTM Neural Network (with MinMax scaling)**
- Same architecture but with scaled input sequences
- Scaling hurt performance significantly — the model struggled to learn
- Test Accuracy: **57.4%**

**Logistic Regression**
- Trained on the full bag-of-words feature matrix (~18,000 features)
- Precision-recall curve maintained 80% precision up to 75% recall
- Test Accuracy: **80.6%** | Precision: 0.81 | Recall: 0.70 | F1: 0.75

**Random Forest**
- 100 estimators trained on the same bag-of-words feature matrix
- Test Accuracy: **79.1%**

---

## Results

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **80.6%** |
| Random Forest | 79.1% |
| LSTM (unscaled) | 76.8% |
| LSTM (scaled) | 57.4% |

The Logistic Regression was the best-performing model. The LSTM's underperformance likely reflects the relatively small dataset size — deep learning models typically require much larger corpora to outperform classical methods on text classification tasks. Scaling the input sequences to the LSTM actively degraded performance, suggesting the token index values carry ordinal meaning the model was relying on.

The confusion matrix revealed the model predicts non-disaster tweets more accurately than disaster tweets, likely due to the class imbalance in the dataset (57% non-disaster).

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python | Core scripting language |
| pandas, numpy | Data manipulation |
| scikit-learn | Logistic Regression, Random Forest, evaluation metrics |
| TensorFlow / Keras | LSTM Neural Network |
| NLTK | Stopword removal, tokenization |
| spaCy | Named Entity Recognition for location extraction |
| matplotlib, seaborn | Data visualization |

---

## Data

This project uses the publicly available Kaggle dataset: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started). The notebook is shared as a **demonstration** — raw data files are not included in this repository.
