# Circuit Vision — Offline
### Hand-drawn circuit schematic → SPICE netlist → Simulation · **No API calls**

Fully local pipeline using YOLOv8 for component detection and OpenCV for wire tracing.
Identical output to the API version — same UI, same netlist format, same grading engine.

---

## Quick start (macOS)

```bash
# 1. Copy your trained weights
cp /path/to/best.pt models/best.pt

# 2. Run — handles venv, deps, opens browser
bash run.sh
```

Then open **http://127.0.0.1:5001**

---

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | `brew install python` |
| Tesseract OCR | `brew install tesseract` |
| YOLOv8 weights | Copy `best.pt` → `models/best.pt` |

No API key needed. No internet required after `pip install`.

---

## Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp /path/to/best.pt models/best.pt
python app.py
```

---

## First-time setup: map your YOLO class names

Your `best.pt` was trained with specific class names. Check what they are:

```bash
python check_classes.py
```

Then open `yolo_classes.json` and make sure every class name from your model
is mapped to a Circuit Vision type string. The full list of valid type strings
is in the `_comment` section at the top of `app.py`.

---

## Pipeline

```
Image upload
    │
    ▼
Pass 1 ── YOLOv8 (best.pt)
           Detects component symbols, returns bboxes + class labels
    │
    ▼
Pass 1b ── Tesseract OCR
           Crops expanded region around each bbox, reads value labels
           (e.g. "10k", "100nF", "5V")
    │
    ▼
Pass 2 ── OpenCV wire tracing
    ├─ Binarise image (Otsu threshold)
    ├─ Mask out component bboxes (leaves wire ends exposed)
    ├─ Skeletonise wires to 1-pixel width (Zhang-Suen thinning)
    ├─ Detect junction pixels (skeleton pixels with ≥3 neighbours)
    ├─ Break skeleton at junctions → isolated wire segments
    ├─ Label connected segments (connectedComponents)
    └─ Match terminal pixels to segment labels → node IDs
    │
    ▼
Stage 2 ── Topology extraction
           Build node objects + edge list from node IDs
    │
    ▼
Stage 3 ── SPICE netlist generation (26 component types)
    │
    ▼
Stage 4 ── Simulation (RC / RL / boost / resistive Euler)
    │
    ▼
Stage 5 ── Overlay rendering (bboxes + node labels)
```

---

## Tuning wire tracing

The wire tracing pipeline has several parameters you can adjust in `app.py`:

| Parameter | Location | Default | Effect |
|-----------|----------|---------|--------|
| `SHRINK` | `assign_nodes_cv` | `0.15` | How much to shrink component bboxes before masking. Increase if wire ends are being cut off. |
| Otsu threshold | `assign_nodes_cv` | auto | Replace with a manual threshold if lighting is uneven. |
| Junction detection `>= 3` | `_find_junctions` | 3 | Lower to 2 to catch more junctions; raises false positives. |
| Terminal search radius | `_terminal_pixels` | ±4px | Increase for lower-res images. |
| `_best_seg` | `assign_nodes_cv` | majority vote | Already robust; tune the search cluster size if needed. |

---

## Grading mode

Identical to the API version. Upload a reference circuit, upload student circuits,
grade individually or in batch. Configurable rubric weights. Export reports as `.txt`.

Grading uses only the topology graph and component data — it does not depend on
the detection method (YOLO or API), so grades are directly comparable.

---

## Project structure

```
circuit-vision-offline/
├── app.py                  ← Full pipeline (YOLO + OpenCV, no API)
├── requirements.txt
├── run.sh                  ← One-command macOS launcher
├── check_classes.py        ← Prints your model's class names
├── yolo_classes.json       ← Maps YOLO class names → CV type strings
├── models/
│   └── best.pt             ← Your trained YOLOv8 weights (copy here)
└── templates/
    └── index.html          ← Identical UI to the API version
```

---

## Differences from the API version

| | API version | Offline version |
|---|---|---|
| Component detection | Claude Sonnet (vision) | YOLOv8 `best.pt` |
| Value reading | Claude Sonnet (OCR in prompt) | Tesseract |
| Node assignment | Claude Sonnet (wire tracing in prompt) | OpenCV skeleton + flood fill |
| Topology / netlist / simulation | Local Python | Identical |
| Grading engine | Local Python | Identical |
| Cost per analysis | ~$0.02–0.03 | $0.00 |
| Works offline | No | Yes |
| Accuracy on messy drawings | Higher | Lower — depends on image quality |

---

## Tips for best accuracy

1. **Draw on white paper** with a **dark pen** — high contrast is critical for Otsu thresholding
2. **Label values clearly** beside each component — OCR reads text near the bbox
3. **Keep wire crossings minimal** — the skeleton tracer can misassign nodes at crossings
4. **Avoid thick ink blobs at junctions** — they can merge into one region and collapse nodes
5. **Photograph straight-on** with even lighting — no shadows, no perspective distortion
