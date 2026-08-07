"""Inference helper for a fine-tuned DistilBERT text classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class DistilBertTextClassifier:
    """Load a trained model and classify transcript text."""

    def __init__(
        self,
        model_dir: str | Path,
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}. "
                "Run `python scripts/train.py` first."
            )

        self.max_length = max_length
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir
        )
        self.model.to(self.device)
        self.model.eval()

    def _label_for_id(self, class_id: int) -> str:
        id2label = self.model.config.id2label
        return str(
            id2label.get(
                class_id,
                id2label.get(str(class_id), f"LABEL_{class_id}"),
            )
        )

    def predict(self, text: str) -> dict[str, Any]:
        """Return the best label, confidence, and probability for every label."""
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("The transcript text cannot be empty.")

        encoded = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}

        with torch.inference_mode():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        predicted_id = int(torch.argmax(probabilities).item())
        scores = {
            self._label_for_id(class_id): round(float(probability), 6)
            for class_id, probability in enumerate(probabilities.cpu())
        }

        return {
            "text": cleaned_text,
            "label": self._label_for_id(predicted_id),
            "confidence": round(float(probabilities[predicted_id].item()), 6),
            "scores": dict(
                sorted(scores.items(), key=lambda item: item[1], reverse=True)
            ),
            "device": str(self.device),
        }
