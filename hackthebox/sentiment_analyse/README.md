# IMDB Sentiment Analysis

This directory contains a binary sentiment classifier for the IMDB movie review dataset. It predicts `1` for positive reviews and `0` for negative reviews.

## Train

From the repository root:

```bash
poetry install
poetry run python sentiment_analyse/train_sentiment_model.py
```

The script trains a scikit-learn pipeline using TF-IDF features and logistic regression, evaluates it on `skills_assessment_data/test.json`, and saves the upload model to:

```text
sentiment_analyse/skills_assessment.joblib
```

Only this one `.joblib` file should be uploaded. The saved model is a fitted sklearn `Pipeline` containing both `TfidfVectorizer` and `LogisticRegression`, trained on raw review text and numeric labels. It predicts `1` for positive and `0` for negative.

Latest test-set result:

```text
Accuracy:  0.9045
Precision: 0.9046
Recall:    0.9044
F1 score:  0.9045
```

## Predict

After training, run:

```bash
poetry run python sentiment_analyse/train_sentiment_model.py --predict "This movie was excellent and moving."
```

The command prints only the predicted label: `1` for positive or `0` for negative.
