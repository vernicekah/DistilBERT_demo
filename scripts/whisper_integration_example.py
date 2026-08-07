"""Example showing how a Whisper transcript can be passed to DistilBERT."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.text_classifier import DistilBertTextClassifier


MODEL_DIR = PROJECT_ROOT / "models" / "distilbert-emergency-classifier"
ALERT_THRESHOLD = 0.80

classifier = DistilBertTextClassifier(MODEL_DIR)


def process_whisper_transcript(transcript: str) -> dict:
    result = classifier.predict(transcript)

    should_alert = (
        result["label"] != "normal"
        and result["confidence"] >= ALERT_THRESHOLD
    )

    if should_alert:
        print(
            f"[ALERT] {result['label']} "
            f"(confidence={result['confidence']:.3f})"
        )
    else:
        print(
            f"[NO ALERT] {result['label']} "
            f"(confidence={result['confidence']:.3f})"
        )

    return result


if __name__ == "__main__":
    # Replace this with the transcript returned by Whisper.
    process_whisper_transcript("help there is a fire in the kitchen")
