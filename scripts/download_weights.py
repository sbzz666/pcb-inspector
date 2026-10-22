"""Download initial weights from the official Ultralytics Hugging Face account."""
import hashlib
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://huggingface.co/Ultralytics/YOLO11/resolve/main/yolo11n.pt?download=true"


def main():
    target = ROOT / "models/yolo11n-official.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print("Weights already present:", target)
        return
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read()
    if not data.startswith(b"PK") or len(data) < 1_000_000:
        raise ValueError("Download did not contain the expected PyTorch archive")
    temporary = target.with_suffix(".part")
    temporary.write_bytes(data)
    temporary.replace(target)
    print("Saved", len(data), "bytes; SHA256", hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
