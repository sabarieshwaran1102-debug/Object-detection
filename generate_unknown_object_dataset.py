from pathlib import Path
import csv
import cv2
import numpy as np

OUTPUT_DIR = Path("unknown_object_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)


def draw_shape(img, shape, color_bgr, center, radius, thickness=2):
    cx, cy = center
    if shape == "circle":
        cv2.circle(img, (cx, cy), radius, color_bgr, thickness=-1)
    elif shape == "square":
        side = radius * 2
        x1 = cx - side // 2
        y1 = cy - side // 2
        cv2.rectangle(img, (x1, y1), (x1 + side, y1 + side), color_bgr, thickness=-1)
    elif shape == "rectangle":
        w_rect = radius * 2
        h_rect = radius * 2 + 30
        x1 = cx - w_rect // 2
        y1 = cy - h_rect // 2
        cv2.rectangle(img, (x1, y1), (x1 + w_rect, y1 + h_rect), color_bgr, thickness=-1)
    elif shape == "triangle":
        pts = np.array([
            [cx, cy - radius],
            [cx - radius, cy + radius],
            [cx + radius, cy + radius]
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], color_bgr)
    elif shape == "hexagon":
        angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) - np.pi / 6
        pts = np.array([
            [cx + int(radius * np.cos(angle)), cy + int(radius * np.sin(angle))]
            for angle in angles
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], color_bgr)
    elif shape == "oval":
        cv2.ellipse(img, (cx, cy), (radius, int(radius * 0.7)), 0, 0, 360, color_bgr, thickness=-1)
    else:
        cv2.ellipse(img, (cx, cy), (radius, radius), 0, 0, 360, color_bgr, thickness=-1)


def create_object_image(filename, color_name, shape, size_label, object_name):
    img = np.full((600, 600, 3), (255, 255, 255), dtype=np.uint8)
    color_map = {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "orange": (0, 165, 255),
        "purple": (128, 0, 128),
        "white": (220, 220, 220),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "cyan": (255, 255, 0),
        "pink": (203, 192, 255),
    }
    size_map = {"tiny": 35, "small": 55, "medium": 85, "large": 120}

    radius = size_map.get(size_label, 70)
    center = (300, 300)
    draw_shape(img, shape, color_map.get(color_name, (0, 0, 0)), center, radius)

    path = OUTPUT_DIR / filename
    cv2.imwrite(str(path), img)
    return path


unknown_objects = [
    {"filename": "yellow_square_small.png", "color": "yellow", "shape": "square", "size": "small", "object_name": "Yellow Square"},
    {"filename": "cyan_hexagon_medium.png", "color": "cyan", "shape": "hexagon", "size": "medium", "object_name": "Cyan Hexagon"},
    {"filename": "black_circle_tiny.png", "color": "black", "shape": "circle", "size": "tiny", "object_name": "Black Circle"},
    {"filename": "purple_rectangle_large.png", "color": "purple", "shape": "rectangle", "size": "large", "object_name": "Purple Rectangle"},
    {"filename": "orange_triangle_small.png", "color": "orange", "shape": "triangle", "size": "small", "object_name": "Orange Triangle"},
    {"filename": "pink_hexagon_medium.png", "color": "pink", "shape": "hexagon", "size": "medium", "object_name": "Pink Hexagon"},
    {"filename": "white_square_small.png", "color": "white", "shape": "square", "size": "small", "object_name": "White Square"},
    {"filename": "blue_triangle_large.png", "color": "blue", "shape": "triangle", "size": "large", "object_name": "Blue Triangle"},
    {"filename": "red_hexagon_medium.png", "color": "red", "shape": "hexagon", "size": "medium", "object_name": "Red Hexagon"},
    {"filename": "green_circle_large.png", "color": "green", "shape": "circle", "size": "large", "object_name": "Green Circle"},
]

manifest_path = OUTPUT_DIR / "unknown_object_manifest.csv"
with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "object_name", "color", "shape", "size"])
    writer.writeheader()
    for obj in unknown_objects:
        create_object_image(obj["filename"], obj["color"], obj["shape"], obj["size"], obj["object_name"])
        writer.writerow({
            "filename": obj["filename"],
            "object_name": obj["object_name"],
            "color": obj["color"],
            "shape": obj["shape"],
            "size": obj["size"],
        })

print(f"Created {len(unknown_objects)} unknown object images in: {OUTPUT_DIR}")
print(f"Manifest: {manifest_path}")
for obj in unknown_objects:
    print(f"- {obj['filename']} -> {obj['object_name']} ({obj['color']}, {obj['shape']}, {obj['size']})")
