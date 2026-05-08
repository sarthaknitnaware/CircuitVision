"""
Run this once to see what class names your best.pt was trained with.
Then update yolo_classes.json to map them to Circuit Vision type strings.

Usage:
    python check_classes.py
"""
from pathlib import Path

MODEL_PATH = Path("models/best.pt")

if not MODEL_PATH.exists():
    print(f"✗  {MODEL_PATH} not found. Copy your weights there first.")
    raise SystemExit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("✗  ultralytics not installed. Run: pip install ultralytics")
    raise SystemExit(1)

model = YOLO(str(MODEL_PATH))
names = model.names   # dict: {int_id: class_name_str}

print(f"\n  best.pt — {len(names)} classes:\n")
for idx, name in sorted(names.items()):
    print(f"    {idx:>3}  {name}")

print("\n  Copy any unmapped names into yolo_classes.json")
