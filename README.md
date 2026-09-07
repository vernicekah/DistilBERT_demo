# DistilBERT Transcript Classifier

This starter project fine-tunes
[`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)
to classify a Whisper transcript into one of six labels:

  - normal
  - fire
  - medical
  - security
  - distress
  - hazard_report


The supplied CSV files are only a small demonstration dataset. Replace and
expand them before using the model in a real safety system.

## Project structure

```text
DISTILBERT/
├── benchmark_results/
│   ├── metrics.json
│   └── predictions.csv
├── config/
│   └── config.yaml
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── modules/
│   ├── __init__.py
│   ├── evaluator.py
│   └── text_classifier.py
├── scripts/
│   ├── train.py
│   ├── predict.py
│   ├── benchmark.py
│   └── download_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `models/` and `benchmark_results/` directories are created automatically
after training and benchmarking.

## 1. Create a virtual environment

From inside the `DISTILBERT` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train the classifier

```bash
python scripts/train.py
```

The first run downloads
[`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)
from Hugging Face. The trained model is saved
to:

```text
models/distilbert-finetuned/
```

## 4. Classify a transcript

```bash
python scripts/predict.py --text "help there is a fire"
```

The program returns JSON similar to:

```json
{
  "label": "fire",
  "confidence": 0.87,
  "scores": {
    "normal": 0.02,
    "fire": 0.87,
    "medical": 0.03,
    "security": 0.02,
    "distress": 0.04,
    "hazard_report": 0.02
  },
  "should_alert": true
}
```

The exact values depend on training.

## 5. Connect it to Whisper

After Whisper produces a transcript, load the trained model with
`DistilBertTextClassifier` and call `predict`:

```python
from modules.text_classifier import DistilBertTextClassifier

classifier = DistilBertTextClassifier("models/distilbert-finetuned")
result = classifier.predict(transcript)
```

`result` is the same dictionary shown in step 4 (`label`, `confidence`,
`scores`, and `should_alert`).

## Important dataset advice

A custom classifier learns from labelled examples rather than from a keyword
lookup table. For better performance:

1. Add many different ways of expressing each event.
2. Include ordinary sentences containing words such as `help` and `fire`.
3. Add negations such as `there is no fire`.
4. Add sentences about drills, movies, and discussions that should be normal.
5. Include punctuation-free and imperfect transcripts similar to Whisper
   output.
6. Keep the number of examples reasonably balanced across labels.
7. Use separate train, validation, and test datasets.
8. Evaluate each class with precision, recall, F1 score, and a confusion
   matrix before deployment.
