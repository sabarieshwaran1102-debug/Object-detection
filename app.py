import os
import glob
import io
import csv
import base64
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS

from core.engine import (
    KnowledgeBaseEngine,
    analyze_image,
    generate_synthetic_image
)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

kb_engine = KnowledgeBaseEngine()

BASE_DIR = Path(__file__).parent.resolve()
DATASETS = {
    "synthetic": BASE_DIR / "synthetic_object_dataset",
    "acquisition": BASE_DIR / "knowledge_acquisition_dataset",
    "unknown": BASE_DIR / "unknown_object_dataset"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/sample_datasets", methods=["GET"])
def get_sample_datasets():
    results = {}
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    for key, folder_path in DATASETS.items():
        if folder_path.exists():
            files = []
            for ext in extensions:
                files.extend(glob.glob(str(folder_path / ext)))
            
            # Format file items
            file_items = []
            for filepath in sorted(files):
                p = Path(filepath)
                file_items.append({
                    "filename": p.name,
                    "rel_path": str(p.relative_to(BASE_DIR)).replace("\\", "/")
                })
            results[key] = file_items
        else:
            results[key] = []
    return jsonify({"success": True, "datasets": results})

@app.route("/api/analyze", methods=["POST"])
def analyze_single_image():
    try:
        data = request.json or {}
        img = None

        if "sample_path" in data and data["sample_path"]:
            rel_path = data["sample_path"].replace("/", os.sep)
            abs_path = BASE_DIR / rel_path
            if not abs_path.exists():
                return jsonify({"success": False, "error": f"File not found: {rel_path}"}), 404
            img = cv2.imread(str(abs_path))

        elif "image_base64" in data and data["image_base64"]:
            b64_data = data["image_base64"]
            if "," in b64_data:
                b64_data = b64_data.split(",")[1]
            img_bytes = base64.b64decode(b64_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        elif "file" in request.files:
            file_storage = request.files["file"]
            file_bytes = file_storage.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "No valid image provided."}), 400

        analysis = analyze_image(img, kb_engine)
        return jsonify({"success": True, "data": analysis})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/batch_analyze", methods=["POST"])
def batch_analyze():
    try:
        data = request.json or {}
        dataset_key = data.get("dataset_key", "synthetic")
        
        target_dir = DATASETS.get(dataset_key, DATASETS["synthetic"])
        if not target_dir.exists():
            return jsonify({"success": False, "error": f"Dataset path {target_dir} not found."}), 404

        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(str(target_dir / ext)))

        image_paths.sort()

        results = []
        recognized_count = 0
        unknown_count = 0
        total_objects = 0

        for path_str in image_paths:
            img = cv2.imread(path_str)
            if img is None:
                continue
            
            filename = Path(path_str).name
            analysis = analyze_image(img, kb_engine)

            for obj in analysis["objects"]:
                total_objects += 1
                reasoning = obj["reasoning"]
                status = reasoning["status"]
                if status in ["RECOGNIZED", "RECOGNIZED_FALLBACK"]:
                    recognized_count += 1
                else:
                    unknown_count += 1

                results.append({
                    "filename": filename,
                    "object_id": obj["id"],
                    "color": obj["symbolic_facts"]["color"],
                    "shape": obj["symbolic_facts"]["shape"],
                    "size": obj["symbolic_facts"]["size"],
                    "recognized_name": reasoning["recognized_name"],
                    "status": reasoning["status"],
                    "confidence": reasoning["confidence"],
                    "matched_rule": reasoning["matched_rule"] or "None",
                    "explanation": reasoning["explanation"]
                })

        summary = {
            "total_images": len(image_paths),
            "total_objects": total_objects,
            "recognized_count": recognized_count,
            "unknown_count": unknown_count,
            "recognition_rate": round((recognized_count / total_objects * 100), 1) if total_objects > 0 else 0
        }

        return jsonify({"success": True, "summary": summary, "results": results})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/knowledge_base", methods=["GET"])
def get_knowledge_base():
    return jsonify({
        "success": True,
        "knowledge_base": kb_engine.knowledge_base,
        "production_rules": kb_engine.production_rules,
        "total_rules": len(kb_engine.production_rules)
    })

@app.route("/api/add_rule", methods=["POST"])
def add_rule():
    try:
        data = request.json or {}
        name = data.get("name")
        color = data.get("color")
        shape = data.get("shape")
        size = data.get("size")
        aliases = data.get("aliases")

        if not all([name, color, shape, size]):
            return jsonify({"success": False, "error": "Name, color, shape, and size are required."}), 400

        kb_engine.add_rule(name, color, shape, size, aliases)
        return jsonify({
            "success": True,
            "message": f"Rule '{name}' added successfully.",
            "total_rules": len(kb_engine.production_rules)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/delete_rule", methods=["POST"])
def delete_rule():
    try:
        data = request.json or {}
        name = data.get("name")
        if not name:
            return jsonify({"success": False, "error": "Rule name is required."}), 400
        
        kb_engine.delete_rule(name)
        return jsonify({
            "success": True,
            "message": f"Rule '{name}' deleted successfully.",
            "total_rules": len(kb_engine.production_rules)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/reset_kb", methods=["POST"])
def reset_kb():
    try:
        kb_engine.reset_knowledge_base()
        return jsonify({
            "success": True,
            "message": "Knowledge base reset to default 13 entries.",
            "total_rules": len(kb_engine.production_rules)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/generate_synthetic", methods=["POST"])
def generate_synthetic():
    try:
        data = request.json or {}
        color = data.get("color", "red")
        shape = data.get("shape", "circle")
        size = data.get("size", "medium")

        synthetic_img = generate_synthetic_image(color, shape, size)
        analysis = analyze_image(synthetic_img, kb_engine)

        return jsonify({"success": True, "data": analysis})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/export_csv", methods=["POST"])
def export_csv():
    try:
        data = request.json or {}
        results = data.get("results", [])

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "filename", "object_id", "color", "shape", "size",
            "recognized_name", "status", "confidence", "matched_rule", "explanation"
        ])
        writer.writeheader()
        writer.writerows(results)

        csv_data = output.getvalue()
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=classical_ai_recognition_report.csv"}
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
