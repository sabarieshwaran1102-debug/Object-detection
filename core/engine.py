import os
import cv2
import numpy as np
import base64
import json
from pathlib import Path

# Base default knowledge base entries from notebook
DEFAULT_KNOWLEDGE_BASE = [
    {"name": "Red Ball", "color": "red", "shape": "circle", "size": "small", "aliases": ["red", "orange"]},
    {"name": "Blue Box", "color": "blue", "shape": "square", "size": "medium", "aliases": ["blue", "cyan"]},
    {"name": "Green Triangle", "color": "green", "shape": "triangle", "size": "small", "aliases": ["green"]},
    {"name": "Yellow Rectangle", "color": "yellow", "shape": "rectangle", "size": "large", "aliases": ["yellow", "orange"]},
    {"name": "Orange Ball", "color": "orange", "shape": "circle", "size": "medium", "aliases": ["orange", "yellow", "red"]},
    {"name": "Purple Box", "color": "purple", "shape": "square", "size": "small", "aliases": ["purple", "blue"]},
    {"name": "White Circle", "color": "white", "shape": "circle", "size": "large", "aliases": ["white", "gray"]},
    {"name": "Black Rectangle", "color": "black", "shape": "rectangle", "size": "medium", "aliases": ["black", "gray"]},
    {"name": "Gray Triangle", "color": "gray", "shape": "triangle", "size": "medium", "aliases": ["gray", "white", "black"]},
    {"name": "Red Rectangle", "color": "red", "shape": "rectangle", "size": "small", "aliases": ["red", "orange"]},
    {"name": "Blue Cylinder", "color": "blue", "shape": "circle", "size": "medium", "aliases": ["blue", "purple"]},
    {"name": "Green Square", "color": "green", "shape": "square", "size": "medium", "aliases": ["green"]},
    {"name": "Cyan Hexagon", "color": "cyan", "shape": "hexagon", "size": "tiny", "aliases": ["cyan", "blue"]}
]

KB_STORAGE_FILE = Path("knowledge_base_store.json")

class KnowledgeBaseEngine:
    def __init__(self):
        self.knowledge_base = []
        self.production_rules = []
        self.load_knowledge_base()

    def load_knowledge_base(self):
        if KB_STORAGE_FILE.exists():
            try:
                with open(KB_STORAGE_FILE, "r") as f:
                    self.knowledge_base = json.load(f)
            except Exception as e:
                print(f"Error loading custom KB store, resetting to default: {e}")
                self.knowledge_base = [dict(item) for item in DEFAULT_KNOWLEDGE_BASE]
        else:
            self.knowledge_base = [dict(item) for item in DEFAULT_KNOWLEDGE_BASE]
        
        self.rebuild_production_rules()

    def save_knowledge_base(self):
        try:
            with open(KB_STORAGE_FILE, "w") as f:
                json.dump(self.knowledge_base, f, indent=2)
        except Exception as e:
            print(f"Error saving KB store: {e}")

    def rebuild_production_rules(self):
        self.production_rules = []
        for entry in self.knowledge_base:
            self.production_rules.append({
                "name": f"{entry['name']} Rule",
                "conditions": {
                    "color": entry["color"],
                    "shape": entry["shape"],
                    "size": entry["size"]
                },
                "result": entry["name"]
            })

    def reset_knowledge_base(self):
        self.knowledge_base = [dict(item) for item in DEFAULT_KNOWLEDGE_BASE]
        self.rebuild_production_rules()
        self.save_knowledge_base()

    def add_rule(self, name, color, shape, size, aliases=None):
        name = name.strip()
        color = color.strip().lower()
        shape = shape.strip().lower()
        size = size.strip().lower()
        
        if size not in {"tiny", "small", "medium", "large"}:
            raise ValueError("Size must be one of: tiny, small, medium, large.")
        
        # Check if exists
        existing = next((item for item in self.knowledge_base if item["name"].casefold() == name.casefold()), None)
        if existing:
            existing["color"] = color
            existing["shape"] = shape
            existing["size"] = size
            if aliases:
                existing["aliases"] = aliases
        else:
            self.knowledge_base.append({
                "name": name,
                "color": color,
                "shape": shape,
                "size": size,
                "aliases": aliases or [color]
            })
        
        self.rebuild_production_rules()
        self.save_knowledge_base()
        return True

    def delete_rule(self, name):
        self.knowledge_base = [item for item in self.knowledge_base if item["name"].casefold() != name.casefold()]
        self.rebuild_production_rules()
        self.save_knowledge_base()
        return True

# Classical Image Processing Helpers

def cv2_to_base64(img_bgr_or_gray):
    """Converts an OpenCV image array to a base64 data URI string."""
    if len(img_bgr_or_gray.shape) == 2:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
    
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

def classify_shape(num_vertices, aspect_ratio, circularity):
    if circularity > 0.8 and num_vertices > 6:
        if 0.85 <= aspect_ratio <= 1.15:
            return "circle"
        return "oval"
    if num_vertices == 3:
        return "triangle"
    if num_vertices == 4:
        if 0.90 <= aspect_ratio <= 1.10:
            return "square"
        return "rectangle"
    if num_vertices == 6:
        return "hexagon"
    return "unknown"

def classify_color(h, s, v):
    if v < 50:
        return "black"
    if s < 40 and v > 200:
        return "white"
    if s < 40:
        return "gray"
    if h >= 170 and s < 120:
        return "pink"
    if h < 10 or h >= 170:
        return "red"
    if 10 <= h < 25:
        return "orange"
    if 25 <= h < 35:
        return "yellow"
    if 35 <= h < 85:
        return "green"
    if 85 <= h < 100:
        return "cyan"
    if 100 <= h < 130:
        return "blue"
    if 130 <= h < 170:
        return "purple"
    return "unknown"

def classify_size(bounding_box, image_shape):
    _, _, width, height = bounding_box
    image_height, image_width = image_shape[:2]
    relative_extent = max(width / image_width, height / image_height)
    
    if relative_extent <= 0.13:
        return "tiny"
    if relative_extent <= 0.25:
        return "small"
    if relative_extent <= 0.34:
        return "medium"
    return "large"

def color_similarity(obs_color, kb_color):
    if obs_color == kb_color:
        return 1.0
    color_groups = {
        "red": {"red", "orange"},
        "orange": {"orange", "yellow", "red"},
        "yellow": {"yellow", "orange", "green"},
        "green": {"green", "yellow", "blue"},
        "blue": {"blue", "cyan", "purple", "green"},
        "cyan": {"cyan", "blue"},
        "purple": {"purple", "blue", "red"},
        "white": {"white", "gray"},
        "black": {"black", "gray", "white"},
        "gray": {"gray", "black", "white"},
    }
    if kb_color in color_groups.get(obs_color, set()) or obs_color in color_groups.get(kb_color, set()):
        return 0.75
    return 0.0

def shape_similarity(obs_shape, kb_shape):
    if obs_shape == kb_shape:
        return 1.0
    if obs_shape in {"square", "rectangle"} and kb_shape in {"square", "rectangle"}:
        return 0.85
    if obs_shape in {"triangle", "rectangle"} and kb_shape in {"triangle", "rectangle"}:
        return 0.55
    return 0.0

def size_similarity(obs_size, kb_size):
    if obs_size == kb_size:
        return 1.0
    size_order = ["tiny", "small", "medium", "large"]
    if obs_size not in size_order or kb_size not in size_order:
        return 0.0
    diff = abs(size_order.index(obs_size) - size_order.index(kb_size))
    return 0.70 if diff == 1 else 0.0

def run_inference(facts, kb_engine):
    kb = kb_engine.knowledge_base
    rules = kb_engine.production_rules
    
    # 1. Primary Check: Exact Production Rule Match
    exact_rule = None
    for rule in rules:
        conds = rule["conditions"]
        if all(facts.get(k) == v for k, v in conds.items()):
            exact_rule = rule
            break

    # 2. Score against all KB entries
    matrix = []
    for entry in kb:
        c_score = color_similarity(facts["color"], entry["color"])
        s_score = shape_similarity(facts["shape"], entry["shape"])
        sz_score = size_similarity(facts["size"], entry["size"])
        weighted = (0.45 * c_score) + (0.35 * s_score) + (0.20 * sz_score)
        
        is_exact = (exact_rule is not None) and (exact_rule["result"] == entry["name"])
        matrix.append({
            "object": entry["name"],
            "rule_name": f"{entry['name']} Rule",
            "exact_match": is_exact,
            "weighted_score": round(weighted, 2),
            "color_score": round(c_score, 2),
            "shape_score": round(s_score, 2),
            "size_score": round(sz_score, 2),
            "kb_color": entry["color"],
            "kb_shape": entry["shape"],
            "kb_size": entry["size"]
        })
    
    # Sort candidates by score descending
    matrix.sort(key=lambda x: x["weighted_score"], reverse=True)

    if exact_rule:
        best = next(m for m in matrix if m["object"] == exact_rule["result"])
        explanation = (
            f"EXACT PRODUCTION RULE MATCH: Triggered '{exact_rule['name']}'. "
            f"Fact matching conditions: color='{facts['color']}', shape='{facts['shape']}', size='{facts['size']}'. "
            f"Confidence: 100% (1.00)."
        )
        return {
            "status": "RECOGNIZED",
            "mode": "EXACT",
            "recognized_name": exact_rule["result"],
            "confidence": 1.0,
            "matched_rule": exact_rule["name"],
            "explanation": explanation,
            "matrix": matrix
        }

    # Fallback to closest weighted match if score >= 0.70
    if matrix and matrix[0]["weighted_score"] >= 0.70:
        best = matrix[0]
        explanation = (
            f"FALLBACK WEIGHTED MATCH: No exact rule fired. Best match is '{best['object']}' "
            f"with similarity score {best['weighted_score']:.2f} (Color: {best['color_score']}, "
            f"Shape: {best['shape_score']}, Size: {best['size_score']})."
        )
        return {
            "status": "RECOGNIZED_FALLBACK",
            "mode": "WEIGHTED_FALLBACK",
            "recognized_name": best["object"],
            "confidence": best["weighted_score"],
            "matched_rule": f"{best['object']} Rule (Fuzzy)",
            "explanation": explanation,
            "matrix": matrix
        }

    # Otherwise Unknown Object
    explanation = (
        f"UNKNOWN OBJECT: Facts (color='{facts['color']}', shape='{facts['shape']}', size='{facts['size']}') "
        f"did not match any known production rules in the Knowledge Base. Highest similarity score was "
        f"{matrix[0]['weighted_score']:.2f} for '{matrix[0]['object']}'."
    )
    return {
        "status": "UNKNOWN",
        "mode": "NONE",
        "recognized_name": "Unknown Object",
        "confidence": matrix[0]["weighted_score"] if matrix else 0.0,
        "matched_rule": None,
        "explanation": explanation,
        "matrix": matrix
    }

def analyze_image(cv_image, kb_engine):
    """Executes the complete classical AI perception and reasoning pipeline."""
    if cv_image is None or cv_image.size == 0:
        raise ValueError("Invalid image input.")
    
    h_img, w_img = cv_image.shape[:2]

    # Stage 1: Original BGR / RGB
    stage_1_b64 = cv2_to_base64(cv_image)

    # Stage 2: Grayscale
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    stage_2_b64 = cv2_to_base64(gray_image)

    # Stage 3: Gaussian Blur Noise Reduction
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    stage_3_b64 = cv2_to_base64(blurred_image)

    # Stage 4: Otsu's Thresholding
    thresh_val, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    stage_4_b64 = cv2_to_base64(binary_image)

    # Stage 5: Morphological Operations (Opening & Closing)
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    stage_5_b64 = cv2_to_base64(cleaned)

    # Stage 6: Contour Detection
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small noise contours
    min_area = max(100, int(0.0005 * h_img * w_img))
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Draw contours and bounding boxes for visualization
    annotated_img = cv_image.copy()
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    detected_objects = []

    for idx, contour in enumerate(valid_contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bounding_box = (x, y, w, h)
        aspect_ratio = float(w) / h if h > 0 else 1.0
        circularity = (4.0 * np.pi * area) / (perimeter ** 2)

        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        # Classifications
        shape = classify_shape(num_vertices, aspect_ratio, circularity)

        # Color classification via HSV mask
        mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_h, mean_s, mean_v, _ = cv2.mean(hsv_image, mask=mask)
        color = classify_color(mean_h, mean_s, mean_v)

        # Size classification
        size = classify_size(bounding_box, cv_image.shape)

        facts = {
            "color": color,
            "shape": shape,
            "size": size
        }

        # Run Production Rule & Inference Engine
        reasoning = run_inference(facts, kb_engine)

        # Draw contour & bounding box on visual output
        color_bgr = (0, 255, 0) if reasoning["status"] != "UNKNOWN" else (0, 0, 255)
        cv2.drawContours(annotated_img, [contour], -1, color_bgr, 2)
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        label = f"#{idx+1}: {reasoning['recognized_name']}"
        cv2.putText(annotated_img, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        detected_objects.append({
            "id": idx + 1,
            "bounding_box": {"x": x, "y": y, "w": w, "h": h},
            "features": {
                "area": round(area, 2),
                "perimeter": round(perimeter, 2),
                "aspect_ratio": round(aspect_ratio, 2),
                "circularity": round(circularity, 2),
                "num_vertices": num_vertices,
                "mean_hsv": [round(mean_h, 1), round(mean_s, 1), round(mean_v, 1)]
            },
            "symbolic_facts": facts,
            "reasoning": reasoning
        })

    stage_6_b64 = cv2_to_base64(annotated_img)

    return {
        "image_dimensions": {"width": w_img, "height": h_img},
        "pipeline_stages": {
            "stage_1_original": stage_1_b64,
            "stage_2_grayscale": stage_2_b64,
            "stage_3_blurred": stage_3_b64,
            "stage_4_otsu_threshold": stage_4_b64,
            "stage_5_morphological": stage_5_b64,
            "stage_6_detected_contours": stage_6_b64,
            "otsu_threshold_value": float(thresh_val)
        },
        "detected_count": len(detected_objects),
        "objects": detected_objects
    }

def generate_synthetic_image(color_name, shape_name, size_name):
    """Generates a synthetic shape image matching the desired color, shape, and size."""
    img_size = 600
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background
    
    # Map size to pixel radius/extent
    size_map = {
        "tiny": 60,
        "small": 120,
        "medium": 180,
        "large": 260
    }
    extent = size_map.get(size_name.lower(), 180)
    center = (img_size // 2, img_size // 2)

    # Map color name to BGR
    color_bgr_map = {
        "red": (0, 0, 220),
        "blue": (220, 0, 0),
        "green": (0, 180, 0),
        "yellow": (0, 220, 220),
        "orange": (0, 140, 255),
        "purple": (180, 0, 180),
        "cyan": (220, 220, 0),
        "pink": (200, 150, 255),
        "white": (245, 245, 245),
        "black": (20, 20, 20),
        "gray": (120, 120, 120)
    }
    bgr = color_bgr_map.get(color_name.lower(), (20, 20, 20))

    shape = shape_name.lower()
    if shape == "circle":
        cv2.circle(canvas, center, extent // 2, bgr, -1)
    elif shape == "square":
        half = extent // 2
        cv2.rectangle(canvas, (center[0] - half, center[1] - half), (center[0] + half, center[1] + half), bgr, -1)
    elif shape == "rectangle":
        w = extent
        h = int(extent * 0.5)
        cv2.rectangle(canvas, (center[0] - w//2, center[1] - h//2), (center[0] + w//2, center[1] + h//2), bgr, -1)
    elif shape == "triangle":
        h = extent
        pts = np.array([
            [center[0], center[1] - h//2],
            [center[0] - h//2, center[1] + h//2],
            [center[0] + h//2, center[1] + h//2]
        ], np.int32)
        cv2.fillPoly(canvas, [pts], bgr)
    elif shape == "hexagon":
        r = extent // 2
        pts = []
        for i in range(6):
            angle = i * np.pi / 3
            px = int(center[0] + r * np.cos(angle))
            py = int(center[1] + r * np.sin(angle))
            pts.append([px, py])
        cv2.fillPoly(canvas, [np.array(pts, np.int32)], bgr)
    else:
        # Default circle
        cv2.circle(canvas, center, extent // 2, bgr, -1)

    return canvas
