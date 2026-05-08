#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Circuit Vision (Offline) — macOS setup & run script
#  Usage:  bash run.sh
# ─────────────────────────────────────────────────────────────
set -e

VENV=".venv"
PYTHON="python3"

echo ""
echo "  ⊙  Circuit Vision (Offline)"
echo "  ─────────────────────────────────────────"

# ── 1. Check Python ────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "  ✗  python3 not found. Install via: brew install python"
  exit 1
fi
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "  ✓  Python $PY_VER"

# ── 2. Check Tesseract ─────────────────────────────────────
if ! command -v tesseract &>/dev/null; then
  echo "  ⚠  Tesseract not found — OCR value reading will be skipped"
  echo "     Install via: brew install tesseract"
else
  echo "  ✓  Tesseract $(tesseract --version 2>&1 | head -1)"
fi

# ── 3. Check best.pt ──────────────────────────────────────
if [ ! -f "models/best.pt" ]; then
  echo ""
  echo "  ✗  models/best.pt not found."
  echo "     Copy your trained YOLOv8 weights to:  models/best.pt"
  echo "     The server will start but detection will return no components."
  echo ""
else
  echo "  ✓  models/best.pt found"
fi

# ── 4. Virtual environment ─────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo ""
  echo "  Creating virtual environment…"
  $PYTHON -m venv $VENV
fi
source "$VENV/bin/activate"

# ── 5. Install dependencies ────────────────────────────────
echo ""
echo "  Installing / verifying dependencies…"
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  ✓  Dependencies ready"

# ── 6. Launch ──────────────────────────────────────────────
echo ""
echo "  ─────────────────────────────────────────"
echo "  Starting server →  http://127.0.0.1:5001"
echo "  Press Ctrl+C to stop"
echo "  ─────────────────────────────────────────"
echo ""

(sleep 1.5 && open http://127.0.0.1:5001) &
python app.py
