"""
Circuit Vision (Offline) — Hand-drawn circuit → Netlist → SPICE Simulation
Fully local pipeline:
  Pass 1 — YOLOv8 (best.pt)  : component detection + bboxes
  Pass 1b— Tesseract OCR      : value reading from bbox regions
  Pass 2 — OpenCV skeleton    : wire skeletonisation → junction detection → flood fill node assignment
  Stage 2 — extract_topology  : build graph from node assignments
  Stage 3 — generate_netlist  : SPICE netlist (26 component types)
  Stage 4 — simulation        : RC / RL / boost / resistive Euler
  Stage 5 — draw_overlay      : annotated image
No API calls. No internet required.
"""

import os, io, base64, json, time, math, re, traceback
from collections import defaultdict, Counter
from pathlib import Path

from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import networkx as nx
from networkx.algorithms import isomorphism

# ── Optional deps — fail gracefully so the server still starts ────────────────
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("⚠  ultralytics not installed — YOLO detection disabled")

try:
    import pytesseract
    from PIL import Image as PILImage
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False
    print("⚠  pytesseract / pillow not installed — OCR value reading disabled")

try:
    from scipy.ndimage import label as scipy_label
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── YOLO model path ────────────────────────────────────────────────────────────
MODEL_PATH      = Path("models/best.pt")
CLASSES_FILE    = Path("yolo_classes.json")
_yolo_model     = None

def _get_yolo():
    global _yolo_model
    if _yolo_model is None and _YOLO_AVAILABLE and MODEL_PATH.exists():
        _yolo_model = YOLO(str(MODEL_PATH))
    return _yolo_model

# ── YOLO class index → Circuit Vision type string ─────────────────────────────
# Loaded from yolo_classes.json so you can edit without touching app.py.
def _load_class_map():
    if CLASSES_FILE.exists():
        raw = json.loads(CLASSES_FILE.read_text())
        return {k.lower(): v for k, v in raw.items() if not k.startswith("_")}
    # Inline fallback if file missing
    return {
        "resistor":"resistor","capacitor":"capacitor","inductor":"inductor",
        "diode":"diode","led":"led","zener":"zener","zener_diode":"zener",
        "schottky":"schottky","tvs":"tvs","bjt":"bjt","transistor":"bjt",
        "npn":"bjt","pnp":"bjt","mosfet":"mosfet","nmos":"mosfet","pmos":"mosfet",
        "voltage_source":"source","battery":"source","dc_source":"source","v_source":"source",
        "current_source":"current_source","i_source":"current_source",
        "ground":"ground","gnd":"ground","vcc":"vcc","vdd":"vcc","power":"vcc",
        "opamp":"opamp","op_amp":"opamp","comparator":"comparator",
        "switch":"switch","sw":"switch","push_button":"push_button","button":"push_button",
        "relay":"relay","transformer":"transformer","fuse":"fuse",
        "crystal":"crystal","xtal":"crystal","voltage_regulator":"voltage_regulator",
        "potentiometer":"potentiometer","pot":"potentiometer",
        "ammeter":"ammeter","voltmeter":"voltmeter","optocoupler":"optocoupler",
    }

YOLO_CLASS_MAP = _load_class_map()

# Component type → SPICE label prefix for auto-numbering
LABEL_PREFIX = {
    "resistor": "R", "potentiometer": "R", "capacitor": "C", "inductor": "L",
    "diode": "D", "led": "D", "zener": "D", "schottky": "D", "tvs": "D",
    "switch": "SW", "push_button": "SW", "relay": "K",
    "bjt": "Q", "mosfet": "M", "voltage_regulator": "VR",
    "opamp": "U", "comparator": "U",
    "source": "V", "current_source": "I",
    "ground": "GND", "vcc": "VCC",
    "transformer": "T", "crystal": "Y", "fuse": "F",
    "ammeter": "AM", "voltmeter": "VM",
}

MULTI_TERMINAL_TYPES = {
    "bjt":               {"base", "collector", "emitter"},
    "mosfet":            {"gate", "drain", "source"},
    "voltage_regulator": {"in", "gnd_adj", "out"},
    "potentiometer":     {"t1", "wiper", "t2"},
    "relay":             {"coil_a", "coil_b", "com", "no"},
    "opamp":             {"in_neg", "in_pos", "vcc", "vee", "out"},
    "comparator":        {"in_neg", "in_pos", "vcc", "vee", "out"},
}

NEEDS_VALUE = {"resistor", "capacitor", "inductor", "source", "transformer",
               "zener", "crystal", "current_source", "potentiometer"}
MODEL_ONLY  = {"bjt", "mosfet", "voltage_regulator", "opamp", "comparator",
               "optocoupler", "diode", "led", "schottky", "tvs"}

REFERENCE_FILE = Path("reference_circuit.json")


# ═════════════════════════════════════════════════════════════════════════════
# Flask routes
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data      = request.get_json()
        image_b64 = data.get("image_b64", "")
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        img_bytes = base64.b64decode(image_b64)
        np_arr    = np.frombuffer(img_bytes, np.uint8)
        img_cv    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img_cv is None:
            return jsonify({"error": "Could not decode image"}), 400

        h, w    = img_cv.shape[:2]
        log_buf = []
        def log(msg, tag="info"):
            log_buf.append({"msg": msg, "tag": tag, "t": time.time()})

        # ── Pass 1: YOLO component detection ──────────────────────────────
        log("Pass 1: YOLO — detecting components…")
        components = run_yolo_detection(img_cv)
        log(f"Detected {len(components)} components", "ok")

        # ── Pass 1b: OCR value reading ─────────────────────────────────────
        log("Pass 1b: OCR — reading component values…")
        components = read_values_ocr(img_cv, components)
        log("OCR complete", "ok")

        # ── Pass 2: OpenCV wire tracing → node assignment ──────────────────
        log("Pass 2: Wire tracing — skeletonise → junctions → flood fill…")
        components = assign_nodes_cv(img_cv, components)
        log("Node assignment complete", "ok")

        # ── Stage 2: Topology ──────────────────────────────────────────────
        log("Stage 2: Building circuit topology…")
        components = merge_phantom_nodes(components)
        nodes, edges, components = extract_topology(components, w, h)
        log(f"Topology: {len(nodes)} nodes, {len(edges)} edges", "ok")

        # ── Stage 3: Netlist ───────────────────────────────────────────────
        log("Stage 3: Generating SPICE netlist…")
        netlist_lines, has_missing, missing_list = generate_netlist(components)
        if has_missing:
            log(f"Missing values: {', '.join(missing_list)}", "warn")
        else:
            log(f"Netlist ready ({len(netlist_lines)} lines)", "ok")

        # ── Stage 4: Simulation ────────────────────────────────────────────
        sim_result = None
        if has_missing:
            log("Stage 4: Simulation skipped — fill in missing values", "warn")
        else:
            log("Stage 4: Transient simulation…")
            try:
                sim_result = run_spice_simulation(components)
                log(f"Simulation done — Vout={sim_result['Vout_avg']:.3f}V  η={sim_result['eff']*100:.1f}%", "ok")
            except Exception as e:
                log(f"Simulation: {e}", "warn")

        # ── Stage 5: Overlay ───────────────────────────────────────────────
        log("Stage 5: Rendering overlay…")
        overlay_b64 = draw_overlay(img_cv, components)
        log("Overlay rendered", "ok")

        adj = build_adjacency_matrix(nodes, components, edges)

        return jsonify({
            "ok":           True,
            "log":          log_buf,
            "components":   components,
            "netlist":      "\n".join(netlist_lines),
            "has_missing":  has_missing,
            "missing_list": missing_list,
            "sim":          sim_result,
            "adjacency":    adj,
            "nodes":        nodes,
            "edges":        edges,
            "overlay_b64":  overlay_b64,
            "ocr_texts":    [],
            "circuit_description": f"{len(components)}-component circuit",
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/resim", methods=["POST"])
def resim():
    try:
        data       = request.get_json()
        components = data.get("components", [])
        edited     = data.get("edited_values", {})
        for c in components:
            if c["label"] in edited:
                c["value"] = edited[c["label"]]
        result = run_spice_simulation(components)
        lines, has_missing, missing = generate_netlist(components)
        return jsonify({"ok": True, "sim": result,
                        "netlist": "\n".join(lines),
                        "has_missing": has_missing})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Grading endpoints (identical to original) ─────────────────────────────────

@app.route("/reload_classes", methods=["POST"])
def reload_classes():
    """Hot-reload yolo_classes.json without restarting the server."""
    global YOLO_CLASS_MAP
    try:
        YOLO_CLASS_MAP = _load_class_map()
        return jsonify({"ok": True, "classes": len(YOLO_CLASS_MAP), "map": YOLO_CLASS_MAP})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_info", methods=["GET"])
def model_info():
    """Return YOLO model class names and current class map."""
    model = _get_yolo()
    yolo_names = dict(model.names) if model else {}
    unmapped = [n for n in yolo_names.values() if n.lower() not in YOLO_CLASS_MAP]
    return jsonify({
        "ok":           True,
        "model_path":   str(MODEL_PATH),
        "model_loaded": model is not None,
        "yolo_classes": yolo_names,
        "class_map":    YOLO_CLASS_MAP,
        "unmapped":     unmapped,
    })
def save_reference():
    try:
        data = request.get_json()
        ref = {
            "components": data.get("components", []),
            "nodes":      data.get("nodes", []),
            "edges":      data.get("edges", []),
            "netlist":    data.get("netlist", ""),
            "sim":        data.get("sim"),
            "saved_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        REFERENCE_FILE.write_text(json.dumps(ref, indent=2))
        return jsonify({"ok": True, "component_count": len(ref["components"])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/load_reference", methods=["GET"])
def load_reference():
    if not REFERENCE_FILE.exists():
        return jsonify({"ok": True, "reference": None})
    try:
        ref = json.loads(REFERENCE_FILE.read_text())
        return jsonify({"ok": True, "reference": ref})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def compare():
    try:
        data    = request.get_json()
        student = data.get("student")
        ref_raw = data.get("reference")
        weights = data.get("weights")
        if not student or not ref_raw:
            return jsonify({"error": "Need student and reference"}), 400
        result = grade_circuits(ref_raw, student, weights=weights)
        return jsonify({"ok": True, **result})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/batch_grade", methods=["POST"])
def batch_grade():
    try:
        data     = request.get_json()
        ref_raw  = data.get("reference")
        students = data.get("students", [])
        weights  = data.get("weights")
        if not ref_raw:
            return jsonify({"error": "No reference provided"}), 400
        results = []
        for s in students:
            name = s.get("name", "Unknown")
            stu  = s.get("result")
            if not stu:
                results.append({"name": name, "error": "No analysis result"})
                continue
            try:
                g = grade_circuits(ref_raw, stu, weights=weights)
                results.append({"name": name, **g})
            except Exception as e:
                results.append({"name": name, "error": str(e)})
        valid = [r for r in results if "total_score" in r]
        avg   = round(sum(r["total_score"] for r in valid) / len(valid), 1) if valid else 0
        dist  = Counter(r["grade"] for r in valid)
        return jsonify({"ok": True, "results": results, "summary": {
            "count": len(students), "graded": len(valid), "avg": avg,
            "highest": max((r["total_score"] for r in valid), default=0),
            "lowest":  min((r["total_score"] for r in valid), default=0),
            "distribution": dict(dist),
        }})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/export_report", methods=["POST"])
def export_report():
    try:
        data  = request.get_json()
        lines = ["=" * 60,
                 "  CIRCUIT VISION (OFFLINE) — GRADE REPORT",
                 f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 "=" * 60]
        if "grade_result" in data:
            lines += _format_single_report(data.get("student_name", "Student"), data["grade_result"])
        elif "batch_results" in data:
            s = data.get("summary", {})
            lines += [f"\n  CLASS SUMMARY",
                      f"  Students graded : {s.get('graded', 0)}",
                      f"  Average score   : {s.get('avg', 0)}",
                      f"  Highest         : {s.get('highest', 0)}",
                      f"  Lowest          : {s.get('lowest', 0)}"]
            dist = s.get("distribution", {})
            lines.append("  Grade dist      : " + "  ".join(f"{k}:{v}" for k, v in sorted(dist.items())))
            lines.append("")
            for r in data["batch_results"]:
                lines.append("-" * 60)
                if "error" in r:
                    lines.append(f"\n  {r['name']}: ERROR — {r['error']}")
                else:
                    lines += _format_single_report(r["name"], r)
        lines.append("")
        return jsonify({"ok": True, "report_text": "\n".join(lines)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# Pass 1 — YOLO detection
# ═════════════════════════════════════════════════════════════════════════════

def run_yolo_detection(img_cv):
    """
    Run YOLOv8 on the image. Returns a list of component dicts with bboxes
    in percentage coordinates (bbox_x_pct, bbox_y_pct, bbox_w_pct, bbox_h_pct).
    Falls back to an empty list if YOLO is unavailable.
    """
    model = _get_yolo()
    if model is None:
        print("⚠  YOLO model not available — returning empty component list")
        return []

    h, w = img_cv.shape[:2]
    results = model(img_cv, verbose=False)[0]

    components = []
    label_counters = defaultdict(int)

    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = model.names[cls_id].lower()
        ctype    = YOLO_CLASS_MAP.get(cls_name, "unknown")
        conf     = float(box.conf[0])

        # YOLO gives xyxy in pixels
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bx = x1 / w;  by = y1 / h
        bw = (x2 - x1) / w;  bh = (y2 - y1) / h

        prefix = LABEL_PREFIX.get(ctype, "U")
        label_counters[prefix] += 1
        label = f"{prefix}{label_counters[prefix]}"

        components.append({
            "type":         ctype,
            "subtype":      "",
            "label":        label,
            "value":        "",
            "model":        "",
            "node_anode":   None,
            "node_cathode": None,
            "terminals":    None,
            "bbox_x_pct":   round(bx, 4),
            "bbox_y_pct":   round(by, 4),
            "bbox_w_pct":   round(max(bw, 0.02), 4),
            "bbox_h_pct":   round(max(bh, 0.02), 4),
            "confidence":   round(conf, 3),
        })

    # Sort top-to-bottom, left-to-right for stable label ordering
    components.sort(key=lambda c: (round(c["bbox_y_pct"] * 10), c["bbox_x_pct"]))
    return components


# ═════════════════════════════════════════════════════════════════════════════
# Pass 1b — OCR value reading
# ═════════════════════════════════════════════════════════════════════════════

# Regex to find component values like 10k, 4.7uF, 100nH, 5V, 1M etc.
VALUE_RE = re.compile(
    r'\b(\d+\.?\d*)\s*([munpkMGTμ]?)\s*(ohm|Ω|F|H|V|A|Hz|W)?\b',
    re.IGNORECASE
)

def read_values_ocr(img_cv, components):
    """
    For each component, crop a region around its bbox (expanded by 60% on each
    side to capture nearby labels), run Tesseract, and extract a value string.
    Skips types that don't need a value.
    """
    if not _OCR_AVAILABLE:
        return components

    h, w = img_cv.shape[:2]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # Upscale for better OCR
    scale = 2.0
    gray_up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for c in components:
        if c["type"] not in NEEDS_VALUE:
            continue
        if c.get("value"):
            continue

        # Expand bbox for label search (labels are usually beside the symbol)
        pad = 0.6
        bx = c["bbox_x_pct"];  by = c["bbox_y_pct"]
        bw = c["bbox_w_pct"];  bh = c["bbox_h_pct"]
        x1 = max(0, int((bx - bw * pad) * w * scale))
        y1 = max(0, int((by - bh * pad) * h * scale))
        x2 = min(int(thresh.shape[1]), int((bx + bw * (1 + pad)) * w * scale))
        y2 = min(int(thresh.shape[0]), int((by + bh * (1 + pad)) * h * scale))

        roi = thresh[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        try:
            text = pytesseract.image_to_string(
                PILImage.fromarray(roi),
                config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789.uμnpkKMGTVAHFΩohm"
            ).strip()
        except Exception:
            continue

        val = _extract_value(text, c["type"])
        if val:
            c["value"] = val

    return components


def _extract_value(text, ctype):
    """Extract and normalise a component value string from raw OCR text."""
    text = text.replace("O", "0").replace("l", "1").replace("μ", "u")
    m = VALUE_RE.search(text)
    if not m:
        # Try just number + multiplier
        m2 = re.search(r'(\d+\.?\d*)\s*([munpkKMGTμ]?)', text)
        if not m2:
            return ""
        num, suf = m2.group(1), m2.group(2).lower()
    else:
        num, suf = m.group(1), m.group(2).lower()

    suf_map = {"k": "k", "m": "m", "u": "u", "μ": "u", "n": "n", "p": "p",
               "g": "G", "t": "T", "M": "M", "": ""}
    suffix = suf_map.get(suf, suf)

    unit_map = {"resistor": "Ω", "capacitor": "F", "inductor": "H",
                "source": "V", "current_source": "A", "zener": "V"}
    unit = unit_map.get(ctype, "")

    return f"{num}{suffix}{unit}" if (num or suffix) else ""


# ═════════════════════════════════════════════════════════════════════════════
# Pass 2 — Wire tracing: skeletonise → junctions → flood fill → node IDs
# ═════════════════════════════════════════════════════════════════════════════

def assign_nodes_cv(img_cv, components):
    """
    Full OpenCV wire tracing pipeline:
      1. Binarise + invert (wires = white on black)
      2. Remove component bounding boxes (mask them out) so only wires remain
      3. Skeletonise wires to 1-pixel width
      4. Detect junction pixels (skeleton pixels with ≥ 3 neighbours)
      5. Break skeleton at junctions to isolate wire segments
      6. Flood-fill each segment to get a unique region ID
      7. Match component terminal positions to region IDs → node numbers
      8. Assign node 0 to any terminal touching a ground component's region
    """
    h, w = img_cv.shape[:2]

    # ── Step 1: binarise ──────────────────────────────────────────────────
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, wire_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── Step 2: mask out component bboxes ─────────────────────────────────
    # Slightly shrink each bbox so terminal ends remain visible
    SHRINK = 0.15
    comp_mask = wire_mask.copy()
    for c in components:
        if c["type"] in ("ground", "vcc"):
            continue
        bx = c["bbox_x_pct"];  by = c["bbox_y_pct"]
        bw = c["bbox_w_pct"];  bh = c["bbox_h_pct"]
        sx = bw * SHRINK;  sy = bh * SHRINK
        x1 = int((bx + sx) * w);  y1 = int((by + sy) * h)
        x2 = int((bx + bw - sx) * w);  y2 = int((by + bh - sy) * h)
        if x2 > x1 and y2 > y1:
            comp_mask[y1:y2, x1:x2] = 0

    # ── Step 3: skeletonise ───────────────────────────────────────────────
    skeleton = _skeletonise(comp_mask)

    # ── Step 4: find junction pixels (≥ 3 neighbours in skeleton) ────────
    junctions = _find_junctions(skeleton)

    # ── Step 5: break skeleton at junctions ──────────────────────────────
    broken = skeleton.copy()
    broken[junctions > 0] = 0

    # ── Step 6: label connected wire segments ─────────────────────────────
    # Re-add junction pixels after labelling (connected component on broken skeleton)
    num_labels, label_img = cv2.connectedComponents(broken, connectivity=8)
    # Assign junction pixels to the nearest non-zero segment label
    label_img = _assign_junction_labels(label_img, junctions, num_labels)

    # ── Step 7: match terminal positions → segment labels → node IDs ──────
    # Build mapping: segment_label → node_id (initially same, ground merges to 0)
    seg_to_node = {}  # filled in below
    gnd_segs    = set()

    # Find which segments touch ground symbols
    for c in components:
        if c["type"] != "ground":
            continue
        bx = c["bbox_x_pct"];  by = c["bbox_y_pct"]
        bw = c["bbox_w_pct"];  bh = c["bbox_h_pct"]
        # Sample several points around the ground symbol's wire lead (top edge)
        pts = _terminal_pixels(c, w, h, "anode")
        for px, py in pts:
            seg = _get_label(label_img, px, py, w, h)
            if seg > 0:
                gnd_segs.add(seg)

    # Assign node IDs to each non-zero segment
    next_node = 1
    for seg in range(1, num_labels + 50):  # +50 for junctions
        if seg in gnd_segs:
            seg_to_node[seg] = 0
        elif seg > 0:
            seg_to_node[seg] = next_node
            next_node += 1

    # ── Step 8: assign nodes to component terminals ────────────────────────
    for c in components:
        ctype = c["type"]
        if ctype == "ground":
            c["node_anode"]   = 0
            c["node_cathode"] = 0
            continue
        if ctype == "vcc":
            # Find the segment at the VCC lead and assign it a high-side node
            pts = _terminal_pixels(c, w, h, "anode")
            seg = _best_seg(label_img, pts, w, h)
            c["node_anode"]   = seg_to_node.get(seg, next_node)
            c["node_cathode"] = 0
            if seg not in seg_to_node:
                seg_to_node[seg] = next_node; next_node += 1
            continue

        if ctype in MULTI_TERMINAL_TYPES:
            term_keys = list(MULTI_TERMINAL_TYPES[ctype])
            terminals = {}
            for key in term_keys:
                pts = _terminal_pixels(c, w, h, key)
                seg = _best_seg(label_img, pts, w, h)
                nid = seg_to_node.get(seg)
                if nid is None:
                    nid = next_node; seg_to_node[seg] = nid; next_node += 1
                terminals[key] = nid
            c["terminals"]    = terminals
            c["node_anode"]   = None
            c["node_cathode"] = None
        else:
            # 2-terminal: anode side and cathode side
            pts_a = _terminal_pixels(c, w, h, "anode")
            pts_k = _terminal_pixels(c, w, h, "cathode")
            seg_a = _best_seg(label_img, pts_a, w, h)
            seg_k = _best_seg(label_img, pts_k, w, h)

            nid_a = seg_to_node.get(seg_a)
            if nid_a is None:
                nid_a = next_node; seg_to_node[seg_a] = nid_a; next_node += 1

            nid_k = seg_to_node.get(seg_k)
            if nid_k is None:
                nid_k = next_node; seg_to_node[seg_k] = nid_k; next_node += 1

            c["node_anode"]   = nid_a
            c["node_cathode"] = nid_k

    return components


# ── Wire tracing helpers ──────────────────────────────────────────────────────

def _skeletonise(binary_img):
    """Zhang-Suen thinning via OpenCV ximgproc, or iterative fallback."""
    try:
        import cv2.ximgproc as xip
        return xip.thinning(binary_img, thinningType=xip.THINNING_ZHANGSUEN)
    except (ImportError, AttributeError):
        pass
    # Fallback: morphological erosion loop (slower but no extra dep)
    img   = binary_img.copy()
    skel  = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded  = cv2.erode(img, kernel)
        dilated = cv2.dilate(eroded, kernel)
        skel   |= (img - dilated)
        img     = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _find_junctions(skel):
    """
    Return a mask of junction pixels: skeleton pixels with ≥ 3 white neighbours.
    Uses a 3x3 convolution count.
    """
    skel_f  = (skel > 0).astype(np.uint8)
    kernel  = np.ones((3, 3), dtype=np.uint8); kernel[1, 1] = 0
    neighbours = cv2.filter2D(skel_f, -1, kernel)
    junctions  = np.zeros_like(skel_f)
    junctions[(skel_f > 0) & (neighbours >= 3)] = 1
    return junctions


def _assign_junction_labels(label_img, junctions, num_labels):
    """
    Junction pixels were removed from the skeleton before labelling.
    Assign each junction pixel to the nearest non-zero segment label
    using a 5x5 neighbourhood search.
    """
    result = label_img.copy()
    jpts   = np.argwhere(junctions > 0)
    for py, px in jpts:
        if result[py, px] != 0:
            continue
        best_lbl = 0
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = py + dy, px + dx
                if 0 <= ny < result.shape[0] and 0 <= nx < result.shape[1]:
                    lbl = result[ny, nx]
                    if lbl > 0:
                        best_lbl = lbl
                        break
            if best_lbl:
                break
        result[py, px] = best_lbl
    return result


def _terminal_pixels(c, w, h, terminal_key):
    """
    Return a list of (px, py) candidate pixels for a given terminal of c,
    estimated from the component's bounding box geometry.
    """
    bx = c["bbox_x_pct"];  by = c["bbox_y_pct"]
    bw = c["bbox_w_pct"];  bh = c["bbox_h_pct"]
    cx = bx + bw / 2;      cy = by + bh / 2
    ctype = c["type"]

    # Terminal position as (frac_x, frac_y) within bbox
    TERM_POS = {
        # 2-terminal
        "anode":   (0.0, 0.5),
        "cathode": (1.0, 0.5),
        # BJT
        "base":      (0.0, 0.5),
        "collector": (0.5, 0.0),
        "emitter":   (0.5, 1.0),
        # MOSFET
        "gate":   (0.0, 0.5),
        "drain":  (0.5, 0.0),
        "source": (0.5, 1.0),
        # Opamp / comparator
        "in_neg": (0.0, 0.35),
        "in_pos": (0.0, 0.65),
        "out":    (1.0, 0.5),
        "vcc":    (0.5, 0.0),
        "vee":    (0.5, 1.0),
        # Voltage regulator
        "in":      (0.0, 0.5),
        "gnd_adj": (0.5, 1.0),
        # Potentiometer
        "t1":    (0.0, 0.5),
        "wiper": (0.5, 1.0),
        "t2":    (1.0, 0.5),
        # Relay
        "coil_a": (0.0, 0.25),
        "coil_b": (0.0, 0.75),
        "com":    (1.0, 0.25),
        "no":     (1.0, 0.75),
    }

    # For vertical components (taller than wide), swap x/y polarity
    is_vertical = bh > bw * 1.3
    if terminal_key == "anode":
        fx, fy = (0.5, 0.0) if is_vertical else (0.0, 0.5)
    elif terminal_key == "cathode":
        fx, fy = (0.5, 1.0) if is_vertical else (1.0, 0.5)
    else:
        fx, fy = TERM_POS.get(terminal_key, (0.5, 0.5))

    # Centre pixel of terminal lead
    px = int((bx + bw * fx) * w)
    py = int((by + bh * fy) * h)

    # Return a small cluster of candidate pixels around the terminal
    pts = []
    for dy in range(-4, 5, 2):
        for dx in range(-4, 5, 2):
            pts.append((
                max(0, min(w - 1, px + dx)),
                max(0, min(h - 1, py + dy))
            ))
    return pts


def _get_label(label_img, px, py, w, h):
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    return int(label_img[py, px])


def _best_seg(label_img, pts, w, h):
    """
    From a list of candidate (px, py) pixels, return the most common
    non-zero segment label. Falls back to 0 if all pixels are background.
    """
    labels = [_get_label(label_img, px, py, w, h) for px, py in pts]
    non_zero = [l for l in labels if l > 0]
    if not non_zero:
        return 0
    return Counter(non_zero).most_common(1)[0][0]


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — Topology (identical to original)
# ═════════════════════════════════════════════════════════════════════════════

def _component_node_pairs(c):
    bx = c["bbox_x_pct"];  by = c["bbox_y_pct"]
    bw = c["bbox_w_pct"];  bh = c["bbox_h_pct"]
    cx = bx + bw / 2;      cy = by + bh / 2
    ctype = c["type"]

    if c["terminals"] is not None:
        terms = c["terminals"]
        if ctype == "bjt":
            return [("base", terms.get("base"), bx, cy),
                    ("collector", terms.get("collector"), cx, by),
                    ("emitter",   terms.get("emitter"),   cx, by + bh)]
        elif ctype == "mosfet":
            return [("gate",   terms.get("gate"),   bx, cy),
                    ("drain",  terms.get("drain"),  cx, by),
                    ("source", terms.get("source"), cx, by + bh)]
        elif ctype in ("opamp", "comparator"):
            return [("in_neg", terms.get("in_neg"), bx, by + bh * 0.35),
                    ("in_pos", terms.get("in_pos"), bx, by + bh * 0.65),
                    ("vcc",    terms.get("vcc"),    cx, by),
                    ("vee",    terms.get("vee"),    cx, by + bh),
                    ("out",    terms.get("out"),    bx + bw, cy)]
        elif ctype == "voltage_regulator":
            return [("in",      terms.get("in"),      bx, cy),
                    ("gnd_adj", terms.get("gnd_adj"), cx, by + bh),
                    ("out",     terms.get("out"),     bx + bw, cy)]
        elif ctype == "potentiometer":
            return [("t1",    terms.get("t1"),    bx, cy),
                    ("wiper", terms.get("wiper"), cx, by + bh),
                    ("t2",    terms.get("t2"),    bx + bw, cy)]
        elif ctype == "relay":
            return [("coil_a", terms.get("coil_a"), bx, by + bh * 0.25),
                    ("coil_b", terms.get("coil_b"), bx, by + bh * 0.75),
                    ("com",    terms.get("com"),    bx + bw, by + bh * 0.25),
                    ("no",     terms.get("no"),     bx + bw, by + bh * 0.75)]
        else:
            return [(k, v, cx, cy) for k, v in terms.items()]
    else:
        return [("anode",   c["node_anode"],   bx,      cy),
                ("cathode", c["node_cathode"], bx + bw, cy)]


def merge_phantom_nodes(components):
    SKIP_TYPES = {"ground", "vcc", "ammeter", "voltmeter"}

    def _get_nodes(c):
        if c.get("terminals"):
            return [v for v in c["terminals"].values() if v is not None]
        return [v for v in [c.get("node_anode"), c.get("node_cathode")] if v is not None]

    def _replace(components, old_id, new_id):
        for c in components:
            if c.get("terminals"):
                c["terminals"] = {k: (new_id if v == old_id else v)
                                  for k, v in c["terminals"].items()}
            else:
                if c.get("node_anode")   == old_id: c["node_anode"]   = new_id
                if c.get("node_cathode") == old_id: c["node_cathode"] = new_id

    for _ in range(20):
        degree  = defaultdict(set)
        for i, c in enumerate(components):
            if c.get("type", "") in SKIP_TYPES:
                continue
            for nid in _get_nodes(c):
                degree[nid].add(i)

        phantoms = [nid for nid, comps in degree.items()
                    if len(comps) == 1 and nid != 0]
        if not phantoms:
            break
        phantoms.sort(reverse=True)
        phantom  = phantoms[0]
        real_nodes = [nid for nid, comps in degree.items()
                      if (len(comps) >= 2 or nid == 0) and nid != phantom]
        if real_nodes:
            target = min(real_nodes, key=lambda n: abs(n - phantom))
        else:
            comp_idx = next(iter(degree[phantom]))
            others   = [n for n in _get_nodes(components[comp_idx]) if n != phantom]
            target   = others[0] if others else 0
        _replace(components, phantom, target)

    all_ids = set()
    for c in components:
        if c.get("type", "") not in SKIP_TYPES:
            for nid in _get_nodes(c):
                all_ids.add(nid)
    non_zero = sorted(n for n in all_ids if n != 0)
    remap    = {0: 0}
    for new_id, old_id in enumerate(non_zero, start=1):
        remap[old_id] = new_id
    for c in components:
        if c.get("terminals"):
            c["terminals"] = {k: remap.get(v, v) for k, v in c["terminals"].items()}
        else:
            if c.get("node_anode")   is not None:
                c["node_anode"]   = remap.get(c["node_anode"],   c["node_anode"])
            if c.get("node_cathode") is not None:
                c["node_cathode"] = remap.get(c["node_cathode"], c["node_cathode"])
    return components


def extract_topology(components, w, h):
    node_pts = defaultdict(list)
    for c in components:
        for term_name, nid, xp, yp in _component_node_pairs(c):
            if nid is None:
                continue
            node_pts[nid].append((xp, yp))
    if not node_pts:
        return [], [], components

    node_objs = []
    for nid, pts in sorted(node_pts.items()):
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        node_objs.append({"id": nid, "x": round(cx, 4), "y": round(cy, 4),
                          "px": int(cx * w), "py": int(cy * h),
                          "label": f"N{nid}" if nid != 0 else "GND"})

    edges, seen_e = [], set()
    def _add_edge(a, b):
        if a is None or b is None or a == b:
            return
        key = (min(a, b), max(a, b))
        if key not in seen_e:
            seen_e.add(key); edges.append(key)

    for c in components:
        pairs    = _component_node_pairs(c)
        node_ids = [p[1] for p in pairs if p[1] is not None]
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                _add_edge(node_ids[i], node_ids[j])

    return node_objs, edges, components


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — Netlist generation (identical to original)
# ═════════════════════════════════════════════════════════════════════════════

def generate_netlist(components):
    lines   = ["* Circuit Vision (Offline) — auto-generated netlist",
               f"* {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    missing = []
    counters = defaultdict(int)

    def _n(c, key):  return c.get(key, 0) or 0
    def _t(c, key, fallback=0):
        if c.get("terminals") and key in c["terminals"]:
            return c["terminals"][key]
        return fallback

    for c in components:
        ctype = c.get("type", "unknown")
        label = c.get("label", "?")
        val   = c.get("value", "").strip()
        model = c.get("model", "").strip()
        sub   = c.get("subtype", "").lower()

        if ctype in NEEDS_VALUE and not val:
            missing.append(label); val = "???"

        if ctype == "resistor":
            counters["R"] += 1
            lines.append(f"R{counters['R']} {_n(c,'node_anode')} {_n(c,'node_cathode')} {val}")
        elif ctype == "potentiometer":
            counters["R"] += 1
            t1, wiper, t2 = _t(c,"t1"), _t(c,"wiper"), _t(c,"t2")
            lines.append(f"* Potentiometer {label} ({val})")
            lines.append(f"R{counters['R']}a {t1} {wiper} {val if val=='???' else val+'/2'}")
            lines.append(f"R{counters['R']}b {wiper} {t2} {val if val=='???' else val+'/2'}")
        elif ctype == "capacitor":
            counters["C"] += 1
            lines.append(f"C{counters['C']} {_n(c,'node_anode')} {_n(c,'node_cathode')} {val} IC=0")
        elif ctype == "inductor":
            counters["L"] += 1
            lines.append(f"L{counters['L']} {_n(c,'node_anode')} {_n(c,'node_cathode')} {val}")
        elif ctype == "source":
            counters["V"] += 1
            src_type = "SIN(0 1 1k)" if "ac" in sub else f"DC {val}"
            lines.append(f"V{counters['V']} {_n(c,'node_anode')} {_n(c,'node_cathode')} {src_type}")
        elif ctype == "current_source":
            counters["I"] += 1
            lines.append(f"I{counters['I']} {_n(c,'node_anode')} {_n(c,'node_cathode')} DC {val}")
        elif ctype == "diode":
            counters["D"] += 1
            lines.append(f"D{counters['D']} {_n(c,'node_anode')} {_n(c,'node_cathode')} DIODE_1N4001")
            if counters["D"] == 1:
                lines.append(".model DIODE_1N4001 D(Is=2.52n Rs=0.568 N=1.752 Cjo=4p Bv=100)")
        elif ctype == "led":
            counters["D"] += 1
            lines.append(f"D{counters['D']} {_n(c,'node_anode')} {_n(c,'node_cathode')} LED_MODEL")
            if counters["D"] == 1:
                lines.append(".model LED_MODEL D(Is=1e-20 N=1.8 Rs=5)")
        elif ctype == "zener":
            counters["D"] += 1
            bv = val if val != "???" else "5.1"
            lines.append(f"D{counters['D']} {_n(c,'node_cathode')} {_n(c,'node_anode')} ZENER_{counters['D']}")
            lines.append(f".model ZENER_{counters['D']} D(Is=1e-14 Rs=1 Bv={bv} Ibv=1m)")
        elif ctype == "schottky":
            counters["D"] += 1
            lines.append(f"D{counters['D']} {_n(c,'node_anode')} {_n(c,'node_cathode')} SCHOTTKY_BAT54")
            if counters["D"] == 1:
                lines.append(".model SCHOTTKY_BAT54 D(Is=200n Rs=0.1 N=1.4 Bv=30)")
        elif ctype == "switch":
            counters["S"] += 1
            lines.append(f"S{counters['S']} {_n(c,'node_anode')} {_n(c,'node_cathode')} ctrl{counters['S']} 0 SW_IDEAL")
            lines.append(".model SW_IDEAL SW(Ron=0.01 Roff=1MEG Vt=0.5 Vh=0.01)")
        elif ctype == "push_button":
            counters["S"] += 1
            lines.append(f"S{counters['S']} {_n(c,'node_anode')} {_n(c,'node_cathode')} ctrl{counters['S']} 0 SW_IDEAL")
        elif ctype == "bjt":
            counters["Q"] += 1
            bjt_model = model if model else ("NPN_GENERIC" if sub != "pnp" else "PNP_GENERIC")
            b, co, e  = _t(c,"base"), _t(c,"collector"), _t(c,"emitter")
            lines.append(f"Q{counters['Q']} {co} {b} {e} {bjt_model}")
            if "NPN_GENERIC" in bjt_model:
                lines.append(".model NPN_GENERIC NPN(Is=1e-14 Bf=200 Br=2 Vaf=100)")
            elif "PNP_GENERIC" in bjt_model:
                lines.append(".model PNP_GENERIC PNP(Is=1e-14 Bf=200 Br=2 Vaf=100)")
        elif ctype == "mosfet":
            counters["M"] += 1
            mos_model = model if model else ("NMOS_GENERIC" if sub != "pmos" else "PMOS_GENERIC")
            g, d, s   = _t(c,"gate"), _t(c,"drain"), _t(c,"source")
            lines.append(f"M{counters['M']} {d} {g} {s} {s} {mos_model}")
            if "NMOS_GENERIC" in mos_model:
                lines.append(".model NMOS_GENERIC NMOS(Vto=2 Kp=20m W=10u L=1u)")
            elif "PMOS_GENERIC" in mos_model:
                lines.append(".model PMOS_GENERIC PMOS(Vto=-2 Kp=10m W=10u L=1u)")
        elif ctype == "voltage_regulator":
            counters["X"] += 1
            vi, gnd, vo = _t(c,"in"), _t(c,"gnd_adj"), _t(c,"out")
            vreg = model if model else "VREG_7805"
            lines.append(f"* Voltage regulator {label} ({vreg})")
            lines.append(f"EVREG{counters['X']} {vo} {gnd} VALUE={{if(V({vi},{gnd})>7, 5, 0)}}")
        elif ctype in ("opamp", "comparator"):
            counters["X"] += 1
            inp, inn = _t(c,"in_pos"), _t(c,"in_neg")
            vcc, vee, out = _t(c,"vcc"), _t(c,"vee"), _t(c,"out")
            oa_model = model if model else "OPAMP_IDEAL"
            lines.append(f"X{counters['X']} {inn} {inp} {vcc} {vee} {out} {oa_model}")
            lines.append(".subckt OPAMP_IDEAL inn inp vcc vee out")
            lines.append("  EOUT out 0 VALUE={LIMIT(1e5*(V(inp)-V(inn)), V(vee)+0.1, V(vcc)-0.1)}")
            lines.append(".ends")
        elif ctype == "transformer":
            counters["L"] += 1
            n = counters["L"]
            lines.append(f"Lp{n} {_n(c,'node_anode')} {_n(c,'node_cathode')} {val}")
            lines.append(f"Ls{n} {_n(c,'node_cathode')+2} {_n(c,'node_cathode')+3} {val}")
            lines.append(f"K{n} Lp{n} Ls{n} 0.999")
        elif ctype == "relay":
            counters["X"] += 1
            ca, cb, com_, no = _t(c,"coil_a"), _t(c,"coil_b"), _t(c,"com"), _t(c,"no")
            lines.append(f"L_RELAY_{counters['X']} {ca} {cb} 100m")
            lines.append(f"S_RELAY_{counters['X']} {com_} {no} {ca} {cb} SW_RELAY")
            lines.append(".model SW_RELAY SW(Ron=0.1 Roff=1MEG Vt=2.0 Vh=0.1)")
        elif ctype == "fuse":
            counters["R"] += 1
            lines.append(f"R{counters['R']} {_n(c,'node_anode')} {_n(c,'node_cathode')} 0.01 * fuse")
        elif ctype == "crystal":
            counters["X"] += 1
            lines.append(f"L_XTAL{counters['X']} {_n(c,'node_anode')} xn{counters['X']} 10m")
            lines.append(f"C_XTAL{counters['X']} xn{counters['X']} {_n(c,'node_cathode')} 50f")
            lines.append(f"R_XTAL{counters['X']} xn{counters['X']} {_n(c,'node_cathode')} 10")
        elif ctype == "ground":
            lines.append(f"* GND at node {_n(c,'node_anode')}")
        elif ctype == "vcc":
            lines.append(f"* VCC supply at node {_n(c,'node_anode')}")
        else:
            lines.append(f"* {label} ({ctype}){(' ' + val) if val and val != '???' else ''}")

    has_missing = bool(missing)
    if not has_missing:
        lines += ["", ".tran 1n 5m 0 500p",
                  ".options RELTOL=1e-3 ABSTOL=1e-12",
                  ".probe V(*) I(V1)", ".end"]
    else:
        lines += ["", "* Simulation directives omitted — fill in ??? values first"]
    return lines, has_missing, missing


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4 — Simulation (identical to original)
# ═════════════════════════════════════════════════════════════════════════════

def _parse_val(s):
    if not s or s in ("???", ""):
        return None
    m = re.match(r"([\d.]+)\s*([munpkMGTμ]?)", s.strip(), re.IGNORECASE)
    if not m:
        return None
    pref = {"m": 1e-3, "u": 1e-6, "μ": 1e-6, "n": 1e-9, "p": 1e-12,
            "k": 1e3,  "M": 1e6,  "G": 1e9,  "T": 1e12}.get(m.group(2), 1.0)
    return float(m.group(1)) * pref


def run_spice_simulation(components):
    Vin = L = C = None
    resistors = []
    for c in components:
        ctype = c.get("type", "")
        v     = _parse_val(c.get("value", ""))
        if   ctype == "source"    and v and Vin is None: Vin = v
        elif ctype == "inductor"  and v and L   is None: L   = v
        elif ctype == "capacitor" and v and C   is None: C   = v
        elif ctype == "resistor"  and v:                 resistors.append(v)
    if Vin is None:
        raise ValueError("No voltage source with a value found")
    R = resistors[0] if resistors else None
    if L and C and R:    return _sim_boost(Vin, L, C, R)
    if R and C and not L: return _sim_rc(Vin, R, C)
    if R and L and not C: return _sim_rl(Vin, R, L)
    if R and not L and not C: return _sim_resistive(Vin, R)
    if not R: R = 1000.0
    if L and not C: C = 10e-6
    if L and C: return _sim_boost(Vin, L, C, R)
    if C: return _sim_rc(Vin, R, C)
    if L: return _sim_rl(Vin, R, L)
    return _sim_resistive(Vin, R)


def _sim_boost(Vin, L, C, R, duty=0.5, freq=50000.0):
    tmax = 5e-3; dt = 1 / (freq * 200)
    N = min(int(tmax / dt), 12000); T = 1 / freq
    time_a, Vout_a, IL_a, Vs_a = [], [], [], []
    il = 0.0; vc = Vin * duty
    for i in range(N):
        t = i * dt; on = (t % T) / T < duty
        Vl = (Vin - vc) if on else -vc
        ic = il - vc / R
        il += (Vl / L) * dt; vc += (ic / C) * dt
        il = max(il, 0.0); vc = max(vc, 0.0)
        time_a.append(t * 1000); Vout_a.append(vc); IL_a.append(il); Vs_a.append(Vin)
    return _package_sim(time_a, Vout_a, IL_a, Vs_a, R, Vin, mode='boost')


def _sim_rc(Vin, R, C):
    tmax = 5 * R * C; dt = tmax / 2000
    N = min(int(tmax / dt), 5000)
    time_a, Vout_a, IL_a, Vs_a = [], [], [], []
    vc = 0.0
    for i in range(N):
        t = i * dt; ic = (Vin - vc) / R; vc += ic / C * dt
        time_a.append(t * 1000); Vout_a.append(vc); IL_a.append(ic); Vs_a.append(Vin)
    return _package_sim(time_a, Vout_a, IL_a, Vs_a, R, Vin, mode='rc')


def _sim_rl(Vin, R, L):
    tau = L / R; tmax = 5 * tau; dt = tmax / 2000
    N   = min(int(tmax / dt), 5000)
    time_a, Vout_a, IL_a, Vs_a = [], [], [], []
    il = 0.0
    for i in range(N):
        t = i * dt; il += (Vin - il * R) / L * dt; vr = il * R
        time_a.append(t * 1000); Vout_a.append(vr); IL_a.append(il); Vs_a.append(Vin)
    return _package_sim(time_a, Vout_a, IL_a, Vs_a, R, Vin, mode='rl')


def _sim_resistive(Vin, R):
    I = Vin / R
    time_a = [t * 0.001 for t in range(100)]
    return _package_sim(time_a, [Vin]*100, [I]*100, [Vin]*100, R, Vin, mode='resistive')


def _package_sim(time_a, Vout_a, IL_a, Vs_a, R, Vin, mode="generic"):
    last  = max(1, int(len(Vout_a) * 0.8))
    vs    = Vout_a[last:]; ils = IL_a[last:]
    Vavg  = float(np.mean(vs));  ILavg = float(np.mean(ils))
    ripI  = float(np.max(ils) - np.min(ils))
    ripV  = float(np.max(vs)  - np.min(vs))
    Pout  = Vavg ** 2 / R if R else 0
    Pin   = Vin * ILavg if ILavg > 0 else 1e-9
    eff   = Pout / Pin
    stride = max(1, len(time_a) // 600)
    return {"time": time_a[::stride], "Vload": Vout_a[::stride],
            "IL": IL_a[::stride], "Vs": Vs_a[::stride],
            "Vout_avg": round(Vavg, 4), "IL_avg": round(ILavg, 5),
            "ripple_I": round(ripI, 5), "Vripple": round(ripV * 1000, 3),
            "Pout": round(Pout, 2), "eff": round(min(eff, 1.0), 4),
            "sim_mode": mode}


# ═════════════════════════════════════════════════════════════════════════════
# Adjacency matrix (identical to original)
# ═════════════════════════════════════════════════════════════════════════════

def build_adjacency_matrix(nodes, components, edges):
    node_ids = [n["id"] for n in nodes]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n   = len(nodes)
    mat = [[0] * n for _ in range(n)]
    for (a, b) in edges:
        i = idx.get(a); j = idx.get(b)
        if i is not None and j is not None:
            mat[i][j] = 1; mat[j][i] = 1
    for c in components:
        all_nodes = []
        if c.get("terminals"):
            all_nodes = list(c["terminals"].values())
        else:
            a = c.get("node_anode"); b = c.get("node_cathode")
            if a is not None: all_nodes.append(a)
            if b is not None: all_nodes.append(b)
        for p in range(len(all_nodes)):
            for q in range(p + 1, len(all_nodes)):
                i = idx.get(all_nodes[p]); j = idx.get(all_nodes[q])
                if i is not None and j is not None and i != j:
                    mat[i][j] = 1; mat[j][i] = 1
    avg = round(sum(sum(r) for r in mat) / n, 3) if n else 0
    return {"nodes": node_ids, "matrix": mat, "avg_deg": avg}


# ═════════════════════════════════════════════════════════════════════════════
# Stage 5 — Overlay (identical to original)
# ═════════════════════════════════════════════════════════════════════════════

COLORS_BGR = {
    "resistor": (160,160,160), "potentiometer": (130,130,190),
    "capacitor": (50,140,240), "inductor": (180,90,155),
    "diode": (224,120,50),     "led": (50,200,180),
    "zener": (180,80,220),     "schottky": (200,100,60),
    "tvs": (160,60,200),       "switch": (50,180,200),
    "push_button": (40,200,210), "relay": (80,160,200),
    "bjt": (90,200,46),        "mosfet": (46,200,100),
    "voltage_regulator": (100,200,80), "opamp": (26,188,155),
    "comparator": (26,155,188), "source": (40,60,220),
    "current_source": (60,80,240), "ground": (80,80,80),
    "vcc": (60,100,240),       "transformer": (190,180,26),
    "crystal": (200,200,60),   "fuse": (200,140,60),
    "optocoupler": (150,200,100), "unknown": (130,130,130),
}

TERM_SHORT = {
    'base':'B','collector':'C','emitter':'E','gate':'G','drain':'D','source':'S',
    'in':'IN','gnd_adj':'GND','out':'OUT','in_neg':'−','in_pos':'+',
    'vcc':'V+','vee':'V−','t1':'1','wiper':'W','t2':'2',
    'coil_a':'CA','coil_b':'CB','com':'COM','no':'NO','anode':'A','cathode':'K',
}

TERMINAL_POSITIONS = {
    "bjt":    {"base":(0.0,0.5),"collector":(0.5,0.0),"emitter":(0.5,1.0)},
    "mosfet": {"gate":(0.0,0.5),"drain":(0.5,0.0),"source":(0.5,1.0)},
    "opamp":  {"in_neg":(0.0,0.35),"in_pos":(0.0,0.65),"out":(1.0,0.5),
               "vcc":(0.5,0.0),"vee":(0.5,1.0)},
    "comparator": {"in_neg":(0.0,0.35),"in_pos":(0.0,0.65),"out":(1.0,0.5),
                   "vcc":(0.5,0.0),"vee":(0.5,1.0)},
    "voltage_regulator": {"in":(0.0,0.5),"gnd_adj":(0.5,1.0),"out":(1.0,0.5)},
    "potentiometer": {"t1":(0.0,0.5),"wiper":(0.5,1.0),"t2":(1.0,0.5)},
    "relay": {"coil_a":(0.0,0.25),"coil_b":(0.0,0.75),"com":(1.0,0.25),"no":(1.0,0.75)},
}


def draw_overlay(img_cv, components):
    out  = img_cv.copy()
    h, w = out.shape[:2]
    scale = max(w, h) / 800.0

    def _px(xp, yp):
        return int(xp * w), int(yp * h)

    # Pass 1 — bounding boxes
    for c in components:
        ctype = c.get("type", "unknown")
        color = COLORS_BGR.get(ctype, (130, 130, 130))
        bx, by, bw, bh = c["bbox_x_pct"], c["bbox_y_pct"], c["bbox_w_pct"], c["bbox_h_pct"]
        x1, y1 = _px(bx, by)
        x2, y2 = _px(bx + bw, by + bh)
        thick  = max(1, int(scale * 1.5))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

    # Pass 2 — labels outside boxes
    for c in components:
        ctype = c.get("type", "unknown")
        color = COLORS_BGR.get(ctype, (130, 130, 130))
        bx, by, bw, bh = c["bbox_x_pct"], c["bbox_y_pct"], c["bbox_w_pct"], c["bbox_h_pct"]
        x1, y1 = _px(bx, by)
        x2, y2 = _px(bx + bw, by + bh)

        label_parts = [c.get("label", "")]
        if c.get("value"):
            label_parts.append(c["value"])

        fs   = max(0.35, min(0.6, scale * 0.55))
        th   = max(1, int(scale))
        lx   = x2 + max(3, int(scale * 4))
        ly   = y1 + max(12, int(scale * 14))
        for i, part in enumerate(label_parts):
            cv2.putText(out, part, (lx, ly + i * int(scale * 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)

        # Terminal nodes for multi-terminal
        if c.get("terminals"):
            tpos = TERMINAL_POSITIONS.get(ctype, {})
            for tname, nid in c["terminals"].items():
                if nid is None:
                    continue
                fp = tpos.get(tname, (0.5, 0.5))
                tx, ty = _px(bx + bw * fp[0], by + bh * fp[1])
                cv2.circle(out, (tx, ty), max(3, int(scale * 3)), color, -1)
                short = TERM_SHORT.get(tname, tname[:2].upper())
                cv2.putText(out, f"{short}:N{nid}", (tx + 4, ty - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.75, color, th, cv2.LINE_AA)
        else:
            # Node dots for 2-terminal
            for key, xp, yp in [("node_anode",   bx,      by + bh / 2),
                                  ("node_cathode", bx + bw, by + bh / 2)]:
                nid = c.get(key)
                if nid is None:
                    continue
                tx, ty = _px(xp, yp)
                cv2.circle(out, (tx, ty), max(3, int(scale * 3)), color, -1)
                cv2.putText(out, f"N{nid}", (tx + 4, ty - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, fs * 0.75, color, th, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


# ═════════════════════════════════════════════════════════════════════════════
# Grading engine (identical to original)
# ═════════════════════════════════════════════════════════════════════════════

GRADE_WEIGHTS = {
    "component_types": 20, "component_counts": 15,
    "component_values": 20, "topology": 30, "simulation": 15,
}
VALUE_TOLERANCE = 0.10


def grade_circuits(ref, student, weights=None):
    effective_weights = dict(GRADE_WEIGHTS)
    if weights:
        total_w = sum(weights.get(k, 0) for k in GRADE_WEIGHTS)
        if total_w > 0:
            effective_weights = {k: round(weights.get(k, 0) * 100 / total_w) for k in GRADE_WEIGHTS}

    feedback = {}
    scores   = {}

    ref_types = Counter(c["type"] for c in ref["components"] if c["type"] not in ("ground","vcc","unknown"))
    stu_types = Counter(c["type"] for c in student["components"] if c["type"] not in ("ground","vcc","unknown"))
    all_types = set(ref_types) | set(stu_types)
    type_score = len(set(ref_types) & set(stu_types)) / max(len(all_types), 1)
    scores["component_types"] = round(type_score * 100)
    feedback["component_types"] = {
        "score": scores["component_types"],
        "missing": [t for t in ref_types if t not in stu_types],
        "extra":   [t for t in stu_types  if t not in ref_types],
        "ref_inventory":     dict(ref_types),
        "student_inventory": dict(stu_types),
    }

    count_hits, count_total, count_detail = 0, 0, {}
    for t in set(ref_types) | set(stu_types):
        rc = ref_types.get(t, 0); sc = stu_types.get(t, 0)
        count_total += 1
        if rc == sc: count_hits += 1; count_detail[t] = "✓"
        else: count_detail[t] = f"ref={rc}, student={sc}"
    scores["component_counts"] = round(count_hits / max(count_total, 1) * 100)
    feedback["component_counts"] = {"score": scores["component_counts"], "detail": count_detail}

    val_hits, val_total, val_detail = 0, 0, []
    NEEDS_VAL = {"resistor","capacitor","inductor","source","current_source",
                 "zener","transformer","crystal","potentiometer"}
    ref_vals = defaultdict(list)
    stu_vals = defaultdict(list)
    for c in ref["components"]:
        if c["type"] in NEEDS_VAL: ref_vals[c["type"]].append(_parse_val(c.get("value","")))
    for c in student["components"]:
        if c["type"] in NEEDS_VAL: stu_vals[c["type"]].append(_parse_val(c.get("value","")))

    for t in ref_vals:
        rv_list = sorted(v for v in ref_vals[t] if v is not None)
        sv_list = sorted(v for v in stu_vals.get(t,[]) if v is not None)
        for rv in rv_list:
            val_total += 1
            match = next((sv for sv in sv_list
                          if abs(sv-rv)/max(abs(rv),1e-12) <= VALUE_TOLERANCE), None)
            if match is not None:
                val_hits += 1; val_detail.append({"type":t,"ref":rv,"student":match,"ok":True})
                sv_list.remove(match)
            else:
                val_detail.append({"type":t,"ref":rv,"student":sv_list[0] if sv_list else None,"ok":False})

    scores["component_values"] = round(val_hits / max(val_total, 1) * 100) if val_total else 100
    feedback["component_values"] = {"score": scores["component_values"],
                                    "detail": val_detail, "total": val_total, "matched": val_hits}

    topo_result = _compare_topology(ref, student)
    scores["topology"] = topo_result["score"]
    feedback["topology"] = topo_result

    sim_result = _compare_simulation(ref, student)
    scores["simulation"] = sim_result["score"]
    feedback["simulation"] = sim_result

    total = sum(scores[k] * effective_weights[k] / 100 for k in effective_weights)
    return {"total_score": round(total), "max_score": 100,
            "weights": effective_weights, "scores": scores,
            "feedback": feedback, "grade": _letter_grade(total)}


def _letter_grade(score):
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def _build_nx_graph(circuit):
    G = nx.MultiGraph()
    for c in circuit.get("components", []):
        ctype = c.get("type", "unknown")
        if ctype in ("ground", "vcc", "ammeter", "voltmeter"):
            continue
        node_ids = (list(c["terminals"].values()) if c.get("terminals")
                    else [c.get("node_anode"), c.get("node_cathode")])
        node_ids = [n for n in node_ids if n is not None]
        for n in node_ids:
            if not G.has_node(n): G.add_node(n)
        for i in range(len(node_ids)):
            for j in range(i+1, len(node_ids)):
                G.add_edge(node_ids[i], node_ids[j], type=ctype)
    return G


def _graph_edit_distance_score(G1, G2):
    try:
        ged = nx.graph_edit_distance(
            G1, G2,
            node_subst_cost=lambda a,b: 0,
            edge_subst_cost=lambda a,b: 0 if a.get("type")==b.get("type") else 1,
            edge_del_cost=lambda a: 1, edge_ins_cost=lambda a: 1, timeout=4)
        if ged is None:
            ged = abs(G1.number_of_edges() - G2.number_of_edges())
        max_ops = max(G1.number_of_edges() + G2.number_of_edges(), 1)
        return max(0, round((1 - ged / max_ops) * 100))
    except Exception:
        return 0


def _compare_topology(ref, student):
    G_ref = _build_nx_graph(ref)
    G_stu = _build_nx_graph(student)
    is_iso = False
    try:
        nm  = isomorphism.categorical_multiedge_match("type", None)
        GM  = isomorphism.GraphMatcher(G_ref, G_stu, edge_match=nm)
        is_iso = GM.is_isomorphic()
    except Exception:
        pass
    score  = 100 if is_iso else _graph_edit_distance_score(G_ref, G_stu)
    method = "exact_isomorphism" if is_iso else "graph_edit_distance"
    return {"score": score, "isomorphic": is_iso, "method": method,
            "ref_nodes": G_ref.number_of_nodes(), "ref_edges": G_ref.number_of_edges(),
            "stu_nodes": G_stu.number_of_nodes(), "stu_edges": G_stu.number_of_edges(),
            "node_diff": abs(G_ref.number_of_nodes()-G_stu.number_of_nodes()),
            "edge_diff": abs(G_ref.number_of_edges()-G_stu.number_of_edges())}


def _compare_simulation(ref, student):
    ref_sim = ref.get("sim"); stu_sim = student.get("sim")
    if not ref_sim or not stu_sim:
        return {"score": 0 if (ref_sim and not stu_sim) else 50,
                "available": False,
                "reason": "Simulation not available for one or both circuits",
                "ref_Vout": ref_sim.get("Vout_avg") if ref_sim else None,
                "stu_Vout": stu_sim.get("Vout_avg") if stu_sim else None}
    r_vout = ref_sim.get("Vout_avg", 0); s_vout = stu_sim.get("Vout_avg", 0)
    r_eff  = ref_sim.get("eff", 0);      s_eff  = stu_sim.get("eff", 0)
    def _pct_err(a, b): return abs(a-b) / max(abs(a), 1e-9)
    vout_err  = _pct_err(r_vout, s_vout)
    eff_err   = _pct_err(r_eff,  s_eff)
    score     = round((max(0,1-vout_err/0.5)*100 + max(0,1-eff_err/0.5)*100) / 2)
    return {"score": score, "available": True,
            "ref_Vout": r_vout, "stu_Vout": s_vout,
            "ref_eff": r_eff,   "stu_eff": s_eff,
            "vout_err_pct": round(vout_err*100,1),
            "eff_err_pct":  round(eff_err*100,1),
            "sim_mode": ref_sim.get("sim_mode","?")}


def _format_single_report(name, g):
    CRIT_LABELS = {"component_types":"Component types","component_counts":"Component counts",
                   "component_values":"Component values","topology":"Topology / wiring",
                   "simulation":"Simulation match"}
    lines = [f"\n  Student : {name}",
             f"  Score   : {g.get('total_score','?')} / 100  (Grade {g.get('grade','?')})", ""]
    weights = g.get("weights", GRADE_WEIGHTS); scores = g.get("scores", {})
    for k, label in CRIT_LABELS.items():
        sc = scores.get(k, 0); w = weights.get(k, 0)
        bar = ("█" * (sc // 10)).ljust(10)
        lines.append(f"  {label:<22} {bar}  {sc:>3}/100  (weight {w}%)")
    fb = g.get("feedback", {})
    inv = fb.get("component_types", {})
    if inv.get("missing"): lines.append(f"  ⚠ Missing: {', '.join(inv['missing'])}")
    if inv.get("extra"):   lines.append(f"  ⚠ Extra: {', '.join(inv['extra'])}")
    for d in [d for d in fb.get("component_values",{}).get("detail",[]) if not d.get("ok")]:
        lines.append(f"  ✗ {d['type']} value: expected {d.get('ref')}, got {d.get('student')}")
    topo = fb.get("topology", {})
    if topo.get("isomorphic"):
        topo_detail = "exact match"
    else:
        topo_detail = f"{topo.get('node_diff',0)} node(s) off, {topo.get('edge_diff',0)} edge(s) off"
    lines.append(f"  {'✓' if topo.get('isomorphic') else '✗'} Topology: {topo_detail}")
    return lines


# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  ⊙  Circuit Vision (Offline)")
    print(f"  YOLO model : {'✓ ' + str(MODEL_PATH) if MODEL_PATH.exists() else '✗ models/best.pt not found'}")
    print(f"  Tesseract  : {'✓' if _OCR_AVAILABLE else '✗ not installed'}")
    print(f"  networkx   : ✓")
    print("\n  http://127.0.0.1:5001\n")
    app.run(debug=True, port=5001)
