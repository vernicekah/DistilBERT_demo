"""Fine-tune DistilBERT for custom transcript classification."""

from __future__ import annotations

import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_csv_dataset(
    csv_path: Path,
    label2id: dict[str, int],
) -> Dataset:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing columns: {sorted(missing_columns)}"
        )

    unknown_labels = sorted(set(dataframe["label"]) - set(label2id))
    if unknown_labels:
        raise ValueError(
            f"{csv_path} contains labels not listed in config.yaml: "
            f"{unknown_labels}"
        )

    dataframe = dataframe.dropna(subset=["text", "label"]).copy()
    dataframe["text"] = dataframe["text"].astype(str).str.strip()
    dataframe = dataframe[dataframe["text"] != ""]
    dataframe["labels"] = dataframe["label"].map(label2id)
    dataframe = dataframe[["text", "labels"]]

    return Dataset.from_pandas(dataframe, preserve_index=False)


def compute_metrics(eval_prediction: Any) -> dict[str, float]:
    logits, true_labels = eval_prediction
    predicted_labels = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(true_labels, predicted_labels)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    config = load_config()

    labels = list(config["labels"])
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    seed = int(config["training"]["seed"])
    set_seed(seed)

    train_path = PROJECT_ROOT / config["data"]["train_file"]
    validation_path = PROJECT_ROOT / config["data"]["validation_file"]
    output_dir = PROJECT_ROOT / config["model"]["output_dir"]

    train_dataset = load_csv_dataset(train_path, label2id)
    validation_dataset = load_csv_dataset(validation_path, label2id)

    model_name = config["model"]["pretrained_name"]
    max_length = int(config["model"]["max_length"])

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize, batched=True)
    validation_dataset = validation_dataset.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(config["training"]["learning_rate"]),
        "per_device_train_batch_size": int(config["training"]["batch_size"]),
        "per_device_eval_batch_size": int(config["training"]["batch_size"]),
        "num_train_epochs": float(config["training"]["epochs"]),
        "weight_decay": float(config["training"]["weight_decay"]),
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 5,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": seed,
    }

    # Transformers renamed evaluation_strategy to eval_strategy.
    argument_names = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in argument_names:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    metrics = trainer.evaluate()
    print("\nValidation metrics:")
    print(json.dumps(metrics, indent=2))

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics_path = output_dir / "validation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved trained model to: {output_dir}")
    print("Run a prediction with:")
    print(
        'python scripts/predict.py --text "help there is a fire"'
    )


if __name__ == "__main__":
    main()
