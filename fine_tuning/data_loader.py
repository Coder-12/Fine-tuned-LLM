import json
from pathlib import Path

def load_data(file_path: str):
    """
    Loads JSON data from a file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
