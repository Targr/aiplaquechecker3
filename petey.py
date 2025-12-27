#!/usr/bin/env python3
import io
import os
import base64
import shutil
import json
from typing import List, Dict, Tuple, Any, Optional
import cv2

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from ultralytics import YOLO
import zipfile
from collections import Counter

# --- Flask App ---
app = Flask(__name__, static_folder="static", static_url_path="/")

# --- Load YOLO model ---
MODEL_PATH = os.path.expanduser("~/Downloads/finetune_beseech_continue/weights/best.pt")
yolo_model = YOLO(MODEL_PATH)

# --- Class names from the YOLO model ---
CLASS_NAMES = [yolo_model.names[i] for i in sorted(yolo_model.names.keys())] if isinstance(yolo_model.names, dict) else list(yolo_model.names)
CLASS_NAME_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# --------- Image Utils ---------
def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def numpy_to_pil(img_rgb: np.ndarray) -> Image.Image:
    return Image.fromarray(img_rgb)

def base64_to_numpy_image(b64_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(b64_str)
    arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode image bytes")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb

def annotate_image_yolo(img: np.ndarray, detections: List[Dict[str, Any]]) -> Image.Image:
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        label_text = f"{det.get('label','')} {det.get('confidence',0):.2f}"
        draw.text((x1, max(0, y1 - 10)), label_text, fill=(255, 255, 0), font=font)
    return pil

def annotate_image_with_boxes(img: np.ndarray, boxes: List[Dict[str, Any]], circle_overlays: Optional[List[Tuple[int,int,int,int]]] = None) -> Image.Image:
    """
    boxes: list of dicts with x1,y1,x2,y2 and optional label/confidence
    circle_overlays: optional list of (cx, cy, r, plate_id) to draw on image
    """
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    # draw circles first (so boxes sit on top)
    if circle_overlays:
        for item in circle_overlays:
            cx, cy, r, pid = item
            # draw circle outline
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(0, 200, 255), width=3)
            draw.text((cx - r + 6, cy - r + 6), f"Plate {pid}", fill=(0,200,255), font=font)
    for b in boxes:
        x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
        label = b.get("label", "")
        conf = b.get("confidence")
        text = f"{label} {conf:.2f}" if conf is not None else label
        if text.strip():
            draw.text((x1, max(0, y1 - 10)), text, fill=(255, 255, 0), font=font)
    return pil

# --------- Helper: Plate detection (circle detection) ---------
def detect_plates_hough(img_rgb: np.ndarray, min_radius_ratio: float = 0.08, max_radius_ratio: float = 0.45, dp: float = 1.2, param1: int = 100, param2: int = 30) -> List[Tuple[int,int,int]]:
    """
    Detect circular plates using HoughCircles.
    Returns list of (cx, cy, radius) sorted by radius desc.
    """
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # equalize + blur to make circles stronger
    img_gray = cv2.equalizeHist(img_gray)
    img_blur = cv2.medianBlur(img_gray, 9)

    h, w = img_gray.shape[:2]
    min_dim = min(h, w)
    minRadius = max(5, int(min_dim * min_radius_ratio))
    maxRadius = max(10, int(min_dim * max_radius_ratio))

    try:
        circles = cv2.HoughCircles(
            img_blur,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=int(min_dim * 0.25),
            param1=param1,
            param2=param2,
            minRadius=minRadius,
            maxRadius=maxRadius
        )
    except Exception:
        circles = None

    out = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # sort by radius desc (largest first)
        circles_sorted = sorted([(int(cx), int(cy), int(r)) for cx, cy, r in circles], key=lambda x: x[2], reverse=True)
        out = circles_sorted
    return out

def detect_plates_contours(img_rgb: np.ndarray, min_area_ratio: float = 0.02) -> List[Tuple[int,int,int]]:
    """
    Alternative contour-based detection for circular-ish plates.
    Returns list of (cx, cy, radius)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert sometimes helps
    th = cv2.bitwise_not(th)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    img_area = h * w
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < img_area * min_area_ratio:
            continue
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        if radius <= 5:
            continue
        # compute circularity
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-8))
        # circularity close to 1 for circle; allow fairly loose threshold
        if circularity > 0.4:
            candidates.append((int(x), int(y), int(radius)))
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    return candidates

# --------- Core YOLO Processing with Deduplication ---------
def process_yolo(image_bytes: bytes, conf_threshold: float) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return [], np.zeros((1,1,3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model.predict(img_rgb, imgsz=640, verbose=False, conf=conf_threshold)[0]
    raw_detections = []
    if hasattr(results, 'boxes') and results.boxes is not None:
        for box, conf, cls_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            width, height = x2 - x1, y2 - y1
            area = max(0, width * height)
            raw_detections.append((x1, y1, x2, y2, width, height, area, float(conf), int(cls_id)))

    if not raw_detections:
        return [], img_rgb

    # deduplicate by removing overlapping boxes with IoU > 0.5 and keeping highest confidence
    def iou(detA, detB):
        ax1, ay1, ax2, ay2 = detA[0], detA[1], detA[2], detA[3]
        bx1, by1, bx2, by2 = detB[0], detB[1], detB[2], detB[3]
        xA = max(ax1, bx1)
        yA = max(ay1, by1)
        xB = min(ax2, bx2)
        yB = min(ay2, by2)
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(1, (ax2-ax1)*(ay2-ay1))
        boxBArea = max(1, (bx2-bx1)*(by2-by1))
        return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

    raw_detections.sort(key=lambda d: d[7], reverse=True)  # sort by confidence desc
    kept = []
    for det in raw_detections:
        if all(iou(det, k) < 0.5 for k in kept):
            kept.append(det)

    # apply area filtering
    img_area = img_rgb.shape[0] * img_rgb.shape[1]
    max_area = img_area * 0.15
    areas = [d[6] for d in kept] if kept else [0]
    mean_area = np.mean(areas)
    std_area = np.std(areas) if len(areas) > 1 else 0.0
    area_cutoff = mean_area + 2 * std_area

    detections = []
    for x1, y1, x2, y2, width, height, area, conf, cls_id in kept:
        if area > max_area:
            continue
        if std_area > 0 and area > area_cutoff:
            continue
        detections.append({
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": float(conf),
            "label": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id),
            "center_x": int((x1 + x2) / 2),
            "center_y": int((y1 + y2) / 2),
            "width": int(width),
            "height": int(height),
            "area": int(area),
        })

    return detections, img_rgb

# --------- Color counting ---------
def count_detections_by_color(img_rgb: np.ndarray, detections: List[Dict[str, Any]], color_map: Dict[str, Tuple[int,int,int]]) -> Dict[str, int]:
    counts = Counter({color: 0 for color in color_map.keys()})
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        h, w = img_rgb.shape[:2]
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = img_rgb[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            continue
        mean_color = crop.mean(axis=(0,1))
        closest_color = min(color_map.keys(), key=lambda c: np.linalg.norm(np.array(color_map[c]) - mean_color))
        counts[closest_color] += 1
    return dict(counts)

# --------- Manual box processing helper ---------
def process_manual_boxes_from_bytes(image_bytes: bytes, boxes: List[Dict[str, Any]], user_colors: List[str]=None) -> Dict[str, Any]:
    """
    boxes: list of dicts with keys x1,y1,x2,y2 and optional label_id/label
    returns counts, annotated image b64, per-box metadata, and color counts if requested
    """
    arr = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"error": "Could not decode image"}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    per_box = []
    detections = []
    h, w = img_rgb.shape[:2]
    for b in boxes:
        x1 = int(max(0, min(w, b.get("x1", 0))))
        y1 = int(max(0, min(h, b.get("y1", 0))))
        x2 = int(max(0, min(w, b.get("x2", x1+1))))
        y2 = int(max(0, min(h, b.get("y2", y1+1))))
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        area = width * height
        label_id = b.get("label_id")
        label = b.get("label") if b.get("label") else (CLASS_NAMES[label_id] if label_id is not None and label_id < len(CLASS_NAMES) else "")
        det = {
            "bbox": [x1, y1, x2, y2],
            "width": width,
            "height": height,
            "area": area,
            "label": label,
            "label_id": label_id
        }
        detections.append(det)
        per_box.append(det)

    annotated_pil = annotate_image_with_boxes(img_rgb, [{"x1":d["bbox"][0],"y1":d["bbox"][1],"x2":d["bbox"][2],"y2":d["bbox"][3],"label":d.get("label","")} for d in detections])
    annotated_b64 = pil_to_base64(annotated_pil)

    color_counts = {}
    if user_colors:
        color_map = {
            "red": (255,0,0),
            "green": (0,255,0),
            "blue": (0,0,255),
            "yellow": (255,255,0),
            "orange": (255,165,0)
        }
        selected_map = {c: color_map[c] for c in user_colors if c in color_map}
        if selected_map:
            # transform detections to the format expected by count_detections_by_color
            dets_for_count = []
            for d in detections:
                dets_for_count.append({"bbox": d["bbox"]})
            color_counts = count_detections_by_color(img_rgb, dets_for_count, selected_map)

    return {
        "manual_count": len(detections),
        "per_box": per_box,
        "annotated_image_base64": annotated_b64,
        "color_counts": color_counts
    }

# --------- Core Processing (now supports multi_plate grouping) ---------
def process_single_image(image_bytes: bytes, params: Dict[str, Any], reference_bytes: List[Tuple[str, bytes]]) -> Dict[str, Any]:
    """
    Returns dictionary. If params.get("multi_plate") is truthy, the response will try to include:
      "plates": [ { plate_id, center, radius, total_features, detections, color_counts, annotated_image_base64 }, ... ]
    Otherwise returns standard:
      "detections", "annotated_image_base64", "original_image_base64", "color_counts"
    """
    conf_threshold = float(params.get("confidence", 0.25))
    multi_plate = bool(params.get("multi_plate", False))

    detections, img_rgb = process_yolo(image_bytes, conf_threshold)
    annotated_default = annotate_image_yolo(img_rgb, detections)
    annotated_b64 = pil_to_base64(annotated_default)

    # also include original image bytes as base64 so frontend can edit original
    try:
        original_pil = Image.fromarray(img_rgb)
        original_b64 = pil_to_base64(original_pil)
    except Exception:
        original_b64 = annotated_b64

    color_counts = {}
    user_colors = params.get("colors")
    selected_map = {}
    if user_colors:
        color_map = {
            "red": (255,0,0),
            "green": (0,255,0),
            "blue": (0,0,255),
            "yellow": (255,255,0),
            "orange": (255,165,0)
        }
        selected_map = {c: color_map[c] for c in user_colors if c in color_map}
        if selected_map:
            color_counts = count_detections_by_color(img_rgb, detections, selected_map)

    # If multi_plate requested, detect plates and group
    if multi_plate:
        # Try Hough first, fallback to contours
        plates = detect_plates_hough(img_rgb)
        if not plates:
            plates = detect_plates_contours(img_rgb)

        # Only keep the largest N circles if many are found (N optional, keep all by default)
        # We'll keep circles reasonably large and not overlapping duplicates:
        filtered_plates = []
        used_centers = []
        for (cx, cy, r) in plates:
            # ignore if near an existing center (duplicate)
            too_close = False
            for (ucx, ucy, ur) in used_centers:
                dist = np.hypot(cx - ucx, cy - ucy)
                if dist < max(10, ur * 0.4):
                    too_close = True
                    break
            if too_close:
                continue
            filtered_plates.append((cx, cy, r))
            used_centers.append((cx, cy, r))

        plates = filtered_plates

        plate_results = []
        # initialize empty list of detections per plate
        for pid, (cx, cy, r) in enumerate(plates, start=1):
            plate_results.append({
                "plate_id": pid,
                "center": [int(cx), int(cy)],
                "radius": int(r),
                "detections": [],
                "total_features": 0,
                "color_counts": {}
            })

        # assign each detection to nearest plate center if inside any circle
        for det in detections:
            px, py = det["center_x"], det["center_y"]
            assigned_idx = None
            min_dist = None
            for i, p in enumerate(plates):
                cx, cy, r = p
                dist = (px - cx) ** 2 + (py - cy) ** 2
                if dist <= (r ** 2):
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        assigned_idx = i
            if assigned_idx is not None:
                plate_results[assigned_idx]["detections"].append(det)

        # fill totals and per-plate annotations & color counts
        for pr in plate_results:
            pr["total_features"] = len(pr["detections"])
            # per-plate color counting
            if selected_map:
                pr["color_counts"] = count_detections_by_color(img_rgb, pr["detections"], selected_map)
            else:
                pr["color_counts"] = {}
            # create an annotated image for this plate: draw plate circle + only detections for plate
            circle_overlay = [(pr["center"][0], pr["center"][1], pr["radius"], pr["plate_id"])]
            boxes_for_draw = []
            for d in pr["detections"]:
                boxes_for_draw.append({"x1": d["bbox"][0], "y1": d["bbox"][1], "x2": d["bbox"][2], "y2": d["bbox"][3], "label": d.get("label",""), "confidence": d.get("confidence",0)})
            annotated_pil = annotate_image_with_boxes(img_rgb, boxes_for_draw, circle_overlays=circle_overlay)
            pr["annotated_image_base64"] = pil_to_base64(annotated_pil)

        # If no plates found, fall back to default single-image output (so frontend still works)
        if not plate_results:
            return {
                "detections": detections,
                "annotated_image_base64": annotated_b64,
                "original_image_base64": original_b64,
                "color_counts": color_counts,
                "plates_detected": 0,
                "plates": []
            }

        return {
            "detections": detections,
            "annotated_image_base64": annotated_b64,
            "original_image_base64": original_b64,
            "color_counts": color_counts,
            "plates_detected": len(plate_results),
            "plates": plate_results
        }

    # default single-plate response
    return {
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "original_image_base64": original_b64,
        "color_counts": color_counts
    }

# --------- Filesystem helpers for supplemental data ---------
BASE_SUPP = os.path.abspath("supplemental")
PENDING_IMAGES = os.path.join(BASE_SUPP, "pending", "images")
PENDING_LABELS = os.path.join(BASE_SUPP, "pending", "labels")
TRAIN_IMAGES = os.path.join(BASE_SUPP, "train", "images")
TRAIN_LABELS = os.path.join(BASE_SUPP, "train", "labels")

def ensure_dirs():
    for p in (PENDING_IMAGES, PENDING_LABELS, TRAIN_IMAGES, TRAIN_LABELS):
        os.makedirs(p, exist_ok=True)
ensure_dirs()

def sanitize_filename(name: str) -> str:
    # remove path components, spaces -> underscores, basic sanitize
    base = os.path.basename(name)
    base = base.replace(" ", "_")
    return "".join(c for c in base if c.isalnum() or c in "._-").strip()

def save_yolo_label_file(label_path: str, boxes: List[Dict[str, Any]], img_w: int, img_h: int):
    """
    Write a YOLO-format .txt: each line: class_id x_center y_center width height (normalized 0..1)
    """
    lines = []
    for b in boxes:
        cls = int(b.get("label_id", 0))
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        # clamp
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        x_center_n = x_center / img_w
        y_center_n = y_center / img_h
        w_n = bw / img_w
        h_n = bh / img_h
        lines.append(f"{cls} {x_center_n:.6f} {y_center_n:.6f} {w_n:.6f} {h_n:.6f}")
    with open(label_path, "w") as fh:
        fh.write("\n".join(lines))

# --------- Routes ---------
@app.route("/")
def root():
    # serve static/index.html from your static folder
    return send_from_directory(app.static_folder, "index.html")

@app.get("/api/classes")
def api_classes():
    return jsonify({"classes": CLASS_NAMES})

@app.post("/api/process")
def api_process():
    if "image" not in request.files:
        return jsonify({"error": "Missing image"}), 400

    image_file = request.files["image"]
    conf_threshold = request.form.get("confidence", 0.25)
    # parse multi_plate flag (allow 'true'/'false' strings)
    mp = request.form.get("multi_plate", "false")
    multi_plate = False
    if isinstance(mp, str):
        multi_plate = mp.lower() in ("1", "true", "yes", "y")
    else:
        multi_plate = bool(mp)

    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    color_param = request.form.get("colors")
    colors = []
    if color_param:
        try:
            # safer parse: accept JSON-list-like strings or comma-separated
            if isinstance(color_param, str):
                color_param = color_param.strip()
                if color_param.startswith('['):
                    colors = json.loads(color_param)
                elif color_param.startswith("'") or color_param.startswith('"'):
                    # try json.loads anyway
                    colors = json.loads(color_param)
                elif ',' in color_param:
                    colors = [c.strip() for c in color_param.split(',') if c.strip()]
                else:
                    # single color
                    colors = [color_param]
            else:
                colors = list(color_param)
        except Exception:
            try:
                # fallback to eval if user passed python list literal
                colors = list(eval(color_param))
            except Exception:
                colors = []

    params = {"confidence": conf_threshold, "colors": colors, "multi_plate": multi_plate}
    out = process_single_image(image_file.read(), params, references)
    return jsonify(out)

@app.post("/api/batch")
def api_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images uploaded"}), 400

    conf_threshold = request.form.get("confidence", 0.25)
    mp = request.form.get("multi_plate", "false")
    multi_plate = False
    if isinstance(mp, str):
        multi_plate = mp.lower() in ("1", "true", "yes", "y")
    else:
        multi_plate = bool(mp)

    ref_files = request.files.getlist("references")
    references: List[Tuple[str, bytes]] = [(rf.filename, rf.read()) for rf in ref_files]

    color_param = request.form.get("colors")
    colors = []
    if color_param:
        try:
            if isinstance(color_param, str):
                color_param = color_param.strip()
                if color_param.startswith('['):
                    colors = json.loads(color_param)
                elif ',' in color_param:
                    colors = [c.strip() for c in color_param.split(',') if c.strip()]
                else:
                    colors = [color_param]
            else:
                colors = list(color_param)
        except Exception:
            try:
                colors = list(eval(color_param))
            except Exception:
                colors = []

    rows = []
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for f in files:
            try:
                image_bytes = f.read()
                result = process_single_image(image_bytes, {"confidence": conf_threshold, "colors": colors, "multi_plate": multi_plate}, references)
                # determine total features (sum across plates if multi_plate)
                if result.get("plates"):
                    total = sum(p.get("total_features", 0) for p in result["plates"])
                    plates_count = len(result["plates"])
                else:
                    total = len(result.get("detections", []))
                    plates_count = 0
                rows.append({
                    "image_name": sanitize_filename(f.filename),
                    "total_features": total,
                    "plates_detected": plates_count,
                    "detections": result.get("detections", []),
                    "plates": result.get("plates", []),
                    "color_counts": result.get("color_counts", {}),
                    "annotated_image_base64": result.get("annotated_image_base64"),
                    "original_image_base64": result.get("original_image_base64")
                })
                # write annotated image(s) into zip; if plates exist write per-plate images, else the single annotated image
                if result.get("plates"):
                    for p in result["plates"]:
                        img_data = base64.b64decode(p["annotated_image_base64"])
                        plate_name = f"{os.path.splitext(sanitize_filename(f.filename))[0]}_plate{p['plate_id']}.png"
                        zf.writestr(plate_name, img_data)
                else:
                    img_data = base64.b64decode(result["annotated_image_base64"])
                    zf.writestr(sanitize_filename(f.filename), img_data)
            except Exception as e:
                rows.append({
                    "image_name": sanitize_filename(f.filename),
                    "error": str(e),
                    "detections": [],
                    "color_counts": {}
                })

    zip_buf.seek(0)
    csv_df = pd.DataFrame(rows)
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    csv_b64 = base64.b64encode(csv_bytes).decode("utf-8")
    zip_b64 = base64.b64encode(zip_buf.read()).decode("utf-8")

    return jsonify({
        "results": rows,
        "csv_base64": csv_b64,
        "zip_base64": zip_b64
    })

@app.post("/api/count_manual")
def api_count_manual():
    """
    Accepts JSON:
    {
      "image_b64": "...",   # base64 PNG/JPG bytes (no data: prefix)
      "boxes": [ {"x1":..,"y1":..,"x2":..,"y2":..,"label_id":0}, ... ],
      "colors": ["red","blue"]   # optional
    }
    Returns:
      manual_count, per_box, annotated_image_base64, color_counts
    """
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    image_b64 = payload.get("image_b64")
    boxes = payload.get("boxes", [])
    colors = payload.get("colors", None)

    if not image_b64:
        return jsonify({"error": "Missing image_b64"}), 400

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid base64 image"}), 400

    try:
        resp = process_manual_boxes_from_bytes(image_bytes, boxes, user_colors=colors)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return jsonify(resp)

@app.post("/api/save_correction")
def api_save_correction():
    """
    Expects JSON:
    {
      "image_name": "...",
      "image_b64": "....",  # base64 PNG (no data: prefix)
      "image_width": 1234,
      "image_height": 987,
      "boxes": [
         {"x1":..,"y1":..,"x2":..,"y2":..,"label_id": 0}, ...
      ],
      "colors": ["red","blue"]  # optional
    }
    Writes:
      supplemental/pending/images/<image_name>  (png)
      supplemental/pending/labels/<basename>.txt (YOLO format)
    Also returns manual counts and annotated image b64 so frontend can immediately display counts.
    """
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON payload"}), 400

    if not payload:
        return jsonify({"error": "Empty payload"}), 400

    image_name = sanitize_filename(payload.get("image_name", "unnamed.png"))
    image_b64 = payload.get("image_b64")
    image_width = int(payload.get("image_width", 0))
    image_height = int(payload.get("image_height", 0))
    boxes = payload.get("boxes", [])
    colors = payload.get("colors", None)

    if not image_b64:
        return jsonify({"error": "Missing image_b64"}), 400

    # ensure extension .png if not present
    if not os.path.splitext(image_name)[1]:
        image_name = image_name + ".png"
    elif os.path.splitext(image_name)[1].lower() not in ('.png', '.jpg', '.jpeg'):
        # force png extension for saved image
        image_name = os.path.splitext(image_name)[0] + ".png"

    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"error": "Invalid base64 image"}), 400

    image_path = os.path.join(PENDING_IMAGES, image_name)
    with open(image_path, "wb") as fh:
        fh.write(image_bytes)

    # YOLO label path
    label_basename = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(PENDING_LABELS, label_basename)
    try:
        if image_width <= 0 or image_height <= 0:
            # try to infer dims from saved image
            img = Image.open(io.BytesIO(image_bytes))
            image_width, image_height = img.size

        save_yolo_label_file(label_path, boxes, image_width, image_height)
    except Exception as e:
        return jsonify({"error": f"Failed to save labels: {str(e)}"}), 500

    # Also compute manual counts and annotated image for immediate frontend feedback
    try:
        manual_resp = process_manual_boxes_from_bytes(image_bytes, boxes, user_colors=colors)
    except Exception as e:
        manual_resp = {"error": f"Failed to compute manual counts: {str(e)}"}

    return jsonify({
        "saved_image": image_path,
        "saved_label_file": label_path,
        "manual_result": manual_resp
    })

@app.post("/api/finalize_supplemental")
def api_finalize_supplemental():
    """
    Move files from supplemental/pending -> supplemental/train
    """
    moved = 0
    # images
    for src_dir, dst_dir in [(PENDING_IMAGES, TRAIN_IMAGES), (PENDING_LABELS, TRAIN_LABELS)]:
        for fname in os.listdir(src_dir):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            try:
                shutil.move(src, dst)
                moved += 1
            except Exception as e:
                app.logger.exception(f"Could not move {src} -> {dst}: {e}")
    return jsonify({"moved": moved})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
