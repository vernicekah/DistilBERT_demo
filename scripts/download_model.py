from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_NAME = "distilbert/distilbert-base-uncased"

SAVE_DIR = Path.home() / "Documents" / "models" / "distilbert-base-uncased"

SAVE_DIR.mkdir(parents=True, exist_ok=True)

downloaded_path = snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=SAVE_DIR,
)

print(f"Original model repository saved to: {downloaded_path}")