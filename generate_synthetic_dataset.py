import os
import csv
from pathlib import Path
import cv2
import numpy as np


OUTPUT_DIR = Path("synthetic_object_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)
KNOWLEDGE_ACQUISITION_DIR = Path("knowledge_acquisition_dataset")


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


def create_object_image(filename, color_name, shape, size_label, output_dir=OUTPUT_DIR):
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

    size_map = {
        "small": 55,
        "medium": 85,
        "large": 120,
        "tiny": 35,
    }

    radius = size_map.get(size_label, 70)
    center = (300, 300)
    draw_shape(img, shape, color_map.get(color_name, (0, 0, 0)), center, radius)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    cv2.imwrite(str(path), img)
    return path


def main():
    objects = [
        ("red_ball_small.png", "red", "circle", "small"),
        ("blue_box_medium.png", "blue", "square", "medium"),
        ("green_triangle_small.png", "green", "triangle", "small"),
        ("yellow_rectangle_large.png", "yellow", "rectangle", "large"),
        ("orange_ball_medium.png", "orange", "circle", "medium"),
        ("purple_box_small.png", "purple", "square", "small"),
        ("white_circle_large.png", "white", "circle", "large"),
        ("black_rectangle_medium.png", "black", "rectangle", "medium"),
        ("gray_triangle_medium.png", "gray", "triangle", "medium"),
        ("red_rectangle_small.png", "red", "rectangle", "small"),
        ("cyan_hexagon_tiny.png", "cyan", "hexagon", "tiny"),
        ("pink_oval_medium.png", "pink", "oval", "medium"),
    ]

    created_files = []
    for filename, color_name, shape, size_label in objects:
        path = create_object_image(filename, color_name, shape, size_label)
        created_files.append(path)

    print(f"Created {len(created_files)} synthetic test images in: {OUTPUT_DIR}")
    for p in created_files:
        print(p)

    new_object_examples = [
        {
            "filename": "orange_triangle_medium.png",
            "color": "orange",
            "shape": "triangle",
            "size": "medium",
            "object_name": "Orange Triangle",
        },
        {
            "filename": "red_hexagon_medium.png",
            "color": "red",
            "shape": "hexagon",
            "size": "medium",
            "object_name": "Red Hexagon",
        },
        {
            "filename": "green_rectangle_large.png",
            "color": "green",
            "shape": "rectangle",
            "size": "large",
            "object_name": "Green Rectangle",
        },
        {
            "filename": "yellow_circle_medium.png",
            "color": "yellow",
            "shape": "circle",
            "size": "medium",
            "object_name": "Yellow Circle",
        },
        {
            "filename": "cyan_triangle_small.png",
            "color": "cyan",
            "shape": "triangle",
            "size": "small",
            "object_name": "Cyan Triangle",
        },
        {
            "filename": "gray_square_large.png",
            "color": "gray",
            "shape": "square",
            "size": "large",
            "object_name": "Gray Square",
        },
        {
            "filename": "black_circle_tiny.png",
            "color": "black",
            "shape": "circle",
            "size": "tiny",
            "object_name": "Black Circle",
        },
        {
            "filename": "white_triangle_medium.png",
            "color": "white",
            "shape": "triangle",
            "size": "medium",
            "object_name": "White Triangle",
        },
        {
            "filename": "pink_rectangle_small.png",
            "color": "pink",
            "shape": "rectangle",
            "size": "small",
            "object_name": "Pink Rectangle",
        },
        {
            "filename": "orange_hexagon_large.png",
            "color": "orange",
            "shape": "hexagon",
            "size": "large",
            "object_name": "Orange Hexagon",
        },
        {
            "filename": "blue_circle_small.png",
            "color": "blue",
            "shape": "circle",
            "size": "small",
            "object_name": "Blue Circle",
        },
        {
            "filename": "green_circle_large.png",
            "color": "green",
            "shape": "circle",
            "size": "large",
            "object_name": "Green Circle",
        },
        {
            "filename": "yellow_square_small.png",
            "color": "yellow",
            "shape": "square",
            "size": "small",
            "object_name": "Yellow Square",
        },
        {
            "filename": "purple_triangle_large.png",
            "color": "purple",
            "shape": "triangle",
            "size": "large",
            "object_name": "Purple Triangle",
        },
        {
            "filename": "cyan_circle_medium.png",
            "color": "cyan",
            "shape": "circle",
            "size": "medium",
            "object_name": "Cyan Circle",
        },
        {
            "filename": "gray_rectangle_small.png",
            "color": "gray",
            "shape": "rectangle",
            "size": "small",
            "object_name": "Gray Rectangle",
        },
        {
            "filename": "black_square_large.png",
            "color": "black",
            "shape": "square",
            "size": "large",
            "object_name": "Black Square",
        },
        {
            "filename": "white_rectangle_medium.png",
            "color": "white",
            "shape": "rectangle",
            "size": "medium",
            "object_name": "White Rectangle",
        },
        {
            "filename": "red_square_large.png",
            "color": "red",
            "shape": "square",
            "size": "large",
            "object_name": "Red Square",
        },
        {
            "filename": "orange_rectangle_small.png",
            "color": "orange",
            "shape": "rectangle",
            "size": "small",
            "object_name": "Orange Rectangle",
        },
        {
            "filename": "pink_circle_large.png",
            "color": "pink",
            "shape": "circle",
            "size": "large",
            "object_name": "Pink Circle",
        },
        {
            "filename": "cyan_square_medium.png",
            "color": "cyan",
            "shape": "square",
            "size": "medium",
            "object_name": "Cyan Square",
        },
        {
            "filename": "blue_hexagon_small.png",
            "color": "blue",
            "shape": "hexagon",
            "size": "small",
            "object_name": "Blue Hexagon",
        },
        {
            "filename": "green_hexagon_large.png",
            "color": "green",
            "shape": "hexagon",
            "size": "large",
            "object_name": "Green Hexagon",
        },
        {
            "filename": "yellow_triangle_small.png",
            "color": "yellow",
            "shape": "triangle",
            "size": "small",
            "object_name": "Yellow Triangle",
        },
        {
            "filename": "purple_rectangle_medium.png",
            "color": "purple",
            "shape": "rectangle",
            "size": "medium",
            "object_name": "Purple Rectangle",
        },
        {
            "filename": "gray_circle_large.png",
            "color": "gray",
            "shape": "circle",
            "size": "large",
            "object_name": "Gray Circle",
        },
        {
            "filename": "black_triangle_small.png",
            "color": "black",
            "shape": "triangle",
            "size": "small",
            "object_name": "Black Triangle",
        },
        {
            "filename": "white_hexagon_large.png",
            "color": "white",
            "shape": "hexagon",
            "size": "large",
            "object_name": "White Hexagon",
        },
        {
            "filename": "red_triangle_medium.png",
            "color": "red",
            "shape": "triangle",
            "size": "medium",
            "object_name": "Red Triangle",
        },
        {
            "filename": "orange_square_large.png",
            "color": "orange",
            "shape": "square",
            "size": "large",
            "object_name": "Orange Square",
        },
        {
            "filename": "cyan_rectangle_large.png",
            "color": "cyan",
            "shape": "rectangle",
            "size": "large",
            "object_name": "Cyan Rectangle",
        },
        {
            "filename": "pink_triangle_medium.png",
            "color": "pink",
            "shape": "triangle",
            "size": "medium",
            "object_name": "Pink Triangle",
        },
        {
            "filename": "blue_triangle_large.png",
            "color": "blue",
            "shape": "triangle",
            "size": "large",
            "object_name": "Blue Triangle",
        },
        {
            "filename": "purple_circle_small.png",
            "color": "purple",
            "shape": "circle",
            "size": "small",
            "object_name": "Purple Circle",
        },
        {
            "filename": "pink_square_large.png",
            "color": "pink",
            "shape": "square",
            "size": "large",
            "object_name": "Pink Square",
        },
    ]

    for example in new_object_examples:
        create_object_image(
            example["filename"],
            example["color"],
            example["shape"],
            example["size"],
            output_dir=KNOWLEDGE_ACQUISITION_DIR,
        )

    manifest_path = KNOWLEDGE_ACQUISITION_DIR / "new_object_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        fieldnames = ["filename", "object_name", "color", "shape", "size"]
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_object_examples)

    print(
        f"\nCreated {len(new_object_examples)} images for knowledge-acquisition demos in: "
        f"{KNOWLEDGE_ACQUISITION_DIR}"
    )
    print(f"New-object manifest: {manifest_path}")


if __name__ == "__main__":
    main()
