"""Benchmark the trained DistilBERT classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.text_classifier import DistilBertTextClassifier
from modules.evaluator import calculate_metrics


def load_config():
    config_path = PROJECT_ROOT / "config" / "config.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main():

    # -----------------------------------
    # 1. Load configuration
    # -----------------------------------

    config = load_config()

    labels = list(config["labels"])

    model_dir = PROJECT_ROOT / config["model"]["output_dir"]

    test_path = PROJECT_ROOT / "data" / "test.csv"

    output_dir = PROJECT_ROOT / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------
    # 2. Load test dataset
    # -----------------------------------

    print(f"Loading test dataset: {test_path}")

    dataframe = pd.read_csv(test_path)

    required_columns = {"text", "label"}

    if not required_columns.issubset(dataframe.columns):
        raise ValueError(
            "test.csv must contain 'text' and 'label' columns."
        )

    # -----------------------------------
    # 3. Load trained DistilBERT
    # -----------------------------------

    print(f"Loading model from: {model_dir}")

    classifier = DistilBertTextClassifier(
        model_dir=model_dir,
        max_length=int(config["model"]["max_length"]),
    )

    predictions = []
    confidences = []
    correct_results = []

    # -----------------------------------
    # 4. Run predictions
    # -----------------------------------

    print("\nRunning benchmark...\n")

    for _, row in dataframe.iterrows():

        text = str(row["text"])
        actual_label = str(row["label"])

        result = classifier.predict(text)

        predicted_label = result["label"]
        confidence = result["confidence"]

        is_correct = predicted_label == actual_label

        predictions.append(predicted_label)
        confidences.append(confidence)
        correct_results.append(is_correct)

        print("-" * 60)
        print(f"Text:       {text}")
        print(f"Actual:     {actual_label}")
        print(f"Predicted:  {predicted_label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Correct:    {is_correct}")

    # -----------------------------------
    # 5. Add predictions to dataframe
    # -----------------------------------

    dataframe["prediction"] = predictions
    dataframe["confidence"] = confidences
    dataframe["correct"] = correct_results

    # -----------------------------------
    # 6. Calculate metrics
    # -----------------------------------

    metrics = calculate_metrics(
        true_labels=dataframe["label"].tolist(),
        predicted_labels=dataframe["prediction"].tolist(),
        labels=labels,
    )

    # -----------------------------------
    # 7. Print overall results
    # -----------------------------------

    print("\n")
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print(
        f"Accuracy:           "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro Precision:    "
        f"{metrics['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall:       "
        f"{metrics['macro_recall']:.4f}"
    )

    print(
        f"Macro F1:           "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1:        "
        f"{metrics['weighted_f1']:.4f}"
    )

    # -----------------------------------
    # 8. Save detailed predictions
    # -----------------------------------

    prediction_path = (
        output_dir / "predictions.csv"
    )

    dataframe.to_csv(
        prediction_path,
        index=False,
    )

    # -----------------------------------
    # 9. Save metrics
    # -----------------------------------

    metrics_path = (
        output_dir / "metrics.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("\nResults saved to:")

    print(f"  {prediction_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    main()