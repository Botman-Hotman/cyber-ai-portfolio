"""Train and use an IMDB sentiment classifier.

The dataset is expected to contain JSON lists with records shaped like:
{"text": "...", "label": 1}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "skills_assessment_data"
DEFAULT_MODEL_PATH = PROJECT_DIR / "sentiment_model.joblib"
UPLOAD_MODEL_PATH = PROJECT_DIR / "skills_assessment.joblib"
STRING_LABEL_MODEL_PATH = PROJECT_DIR / "skills_assessment_string_labels.joblib"
MODEL_ALIASES = (
    PROJECT_DIR / "sentiment_analysis_model.joblib",
    PROJECT_DIR / "sentiment_analyse_model.joblib",
    PROJECT_DIR / "imdb_sentiment_model.joblib",
    PROJECT_DIR / "model.joblib",
)


def load_json_dataset(path: Path) -> tuple[list[str], list[int]]:
    """Load review texts and binary labels from a JSON dataset file."""
    with path.open("r", encoding="utf-8") as file:
        records: list[dict[str, Any]] = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list of records.")

    texts: list[str] = []
    labels: list[int] = []
    for index, record in enumerate(records):
        text = record.get("text")
        label = record.get("label")
        if not isinstance(text, str) or label not in {0, 1}:
            raise ValueError(
                f"{path} record {index} must contain string 'text' and binary 'label'."
            )
        texts.append(text)
        labels.append(int(label))

    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(C=4.0, max_iter=1000, random_state=42),
            ),
        ]
    )


def train_and_evaluate(
    train_path: Path,
    test_path: Path,
    model_path: Path,
) -> None:
    x_train, y_train = load_json_dataset(train_path)
    x_test, y_test = load_json_dataset(test_path)

    print(f"Loaded {len(x_train):,} training reviews from {train_path}")
    print(f"Loaded {len(x_test):,} test reviews from {test_path}")
    print(f"Training label counts: {label_counts(y_train)}")
    print(f"Test label counts: {label_counts(y_test)}")

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    print("\nEvaluation on test set")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions):.4f}")
    print(f"F1 score:  {f1_score(y_test, predictions):.4f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report")
    print(classification_report(y_test, predictions, target_names=["negative", "positive"]))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

    for alias_path in MODEL_ALIASES:
        if alias_path != model_path:
            joblib.dump(pipeline, alias_path)
            print(f"Saved compatibility copy to {alias_path}")

    string_label_pipeline = build_pipeline()
    string_label_pipeline.fit(x_train, to_platform_labels(y_train))
    joblib.dump(string_label_pipeline, UPLOAD_MODEL_PATH)
    joblib.dump(string_label_pipeline, STRING_LABEL_MODEL_PATH)
    string_label_predictions = string_label_pipeline.predict(x_test)
    string_label_accuracy = accuracy_score(to_platform_labels(y_test), string_label_predictions)
    print(f"Saved platform upload model to {UPLOAD_MODEL_PATH}")
    print(f"Saved string-label copy to {STRING_LABEL_MODEL_PATH}")
    print(f"Platform-label accuracy: {string_label_accuracy:.4f}")


def label_counts(labels: list[int]) -> dict[int, int]:
    counts = Counter(labels)
    return {label: counts[label] for label in sorted(counts)}


def to_platform_labels(labels: list[int]) -> list[str]:
    return ["Positive" if label == 1 else "Negative" for label in labels]


def predict_review(model_path: Path, review: str) -> None:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with this script."
        )

    pipeline: Pipeline = joblib.load(model_path)
    raw_prediction = pipeline.predict([review])[0]
    prediction = normalize_prediction(raw_prediction)
    print(prediction)


def normalize_prediction(prediction: Any) -> int:
    if prediction in {1, "1", "Positive", "positive"}:
        return 1
    if prediction in {0, "0", "Negative", "negative"}:
        return 0
    raise ValueError(f"Unexpected prediction label: {prediction!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or use an IMDB sentiment model.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=DATA_DIR / "train.json",
        help="Path to the training JSON file.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=DATA_DIR / "test.json",
        help="Path to the test JSON file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save or load the trained joblib model.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Predict sentiment for this review text instead of training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predict:
        predict_review(args.model_path, args.predict)
    else:
        train_and_evaluate(args.train_path, args.test_path, args.model_path)


if __name__ == "__main__":
    main()
