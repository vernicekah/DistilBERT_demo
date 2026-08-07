"""Classify one transcript using the fine-tuned DistilBERT model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.text_classifier import DistilBertTextClassifier


def load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a transcript using DistilBERT."
    )
    parser.add_argument(
        "--text",
        required=True,
        help='Transcript to classify, for example: "help there is a fire"',
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Optional path to a trained model directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()

    model_dir = (
        Path(args.model_dir)
        if args.model_dir
        else PROJECT_ROOT / config["model"]["output_dir"]
    )

    classifier = DistilBertTextClassifier(
        model_dir=model_dir,
        max_length=int(config["model"]["max_length"]),
    )
    result = classifier.predict(args.text)

    threshold = float(config["inference"]["alert_threshold"])
    result["should_alert"] = (
        result["label"] != "normal"
        and result["confidence"] >= threshold
    )
    result["alert_threshold"] = threshold

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
