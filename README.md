# DistilBERT Transcript Classifier

This starter project fine-tunes `distilbert-base-uncased` to classify a
Whisper transcript into one of four labels:

- `normal`
- `fire`
- `medical_emergency`
- `security_threat`

The supplied CSV files are only a small demonstration dataset. Replace and
expand them before using the model in a real safety system.

## Project structure

```text
DISTILBERT/
├── config/
│   └── config.yaml
├── data/
│   ├── train.csv
│   └── validation.csv
├── modules/
│   ├── __init__.py
│   └── text_classifier.py
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── whisper_integration_example.py
├── requirements.txt
├── .gitignore
└── README.md
```

The `models/` directory is created automatically after training.

## 1. Create a virtual environment

From inside the `DISTILBERT` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
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

The first run downloads `distilbert-base-uncased`. The trained model is saved
to:

```text
models/distilbert-emergency-classifier/
```

## 4. Classify a transcript

```bash
python scripts/predict.py --text "help there is a fire"
```

The program returns JSON similar to:

```json
{
  "label": "fire",
  "confidence": 0.93,
  "scores": {
    "fire": 0.93,
    "medical_emergency": 0.03,
    "security_threat": 0.02,
    "normal": 0.02
  },
  "should_alert": true
}
```

The exact values depend on training.

## 5. Connect it to Whisper

After Whisper produces a transcript:

```python
result = classifier.predict(transcript)
```

See `scripts/whisper_integration_example.py` for a complete example.

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
