# Project Documentation

## Overview

This project is a classical AI object recognition system that combines computer vision preprocessing with a symbolic knowledge base and production-rule inference engine. Instead of relying on deep-learning models, it extracts visual facts from images such as color, shape, and size, then matches them against known object definitions.

The application is built with Python, OpenCV, NumPy, and Flask. It exposes both a web dashboard and REST endpoints for analyzing single images, batch datasets, and adding or updating knowledge-base rules.

## Purpose

The system is designed to:

- identify objects in images using deterministic feature extraction
- classify visual attributes into symbolic facts
- compare observations against a rule base
- recognize known objects with exact matching or fuzzy fallback
- flag unknown combinations when no reliable match is found
- allow knowledge acquisition by editing or extending the rule base

## System Architecture

The project follows a layered architecture in which client requests are translated into symbolic facts and then evaluated by a rule-based reasoning engine.

```mermaid
flowchart TD
    A[User / Browser] --> B[Flask Web App\napp.py]
    B --> C[REST API Layer\n/analyze, /batch_analyze, /knowledge_base, /add_rule]
    C --> D[KnowledgeBaseEngine\ncore/engine.py]
    D --> E[Knowledge Base Store\nknowledge_base_store.json]

    D --> F[Image Acquisition\nfile upload / sample path / base64]
    F --> G[Computer Vision Pipeline]
    G --> G1[Grayscale Conversion]
    G1 --> G2[Gaussian Blur]
    G2 --> G3[Otsu Thresholding]
    G3 --> G4[Morphological Filtering]
    G4 --> G5[Contour Detection]
    G5 --> H[Feature Extraction]

    H --> H1[Color Classification\nHSV analysis]
    H --> H2[Shape Classification\nvertices, aspect ratio, circularity]
    H --> H3[Size Classification\nbounding-box ratio]

    H1 --> I[Symbolic Facts\n{color, shape, size}]
    H2 --> I
    H3 --> I

    I --> J[Production Rule Matcher\nrun_inference]
    D --> J
    E --> J

    J --> K{Exact rule match?}
    K -- Yes --> L[RECOGNIZED\nconfidence = 1.0]
    K -- No --> M{Weighted fuzzy match >= 0.70?}
    M -- Yes --> N[RECOGNIZED_FALLBACK]
    M -- No --> O[UNKNOWN OBJECT]

    L --> P[Response JSON / CSV Export]
    N --> P
    O --> P
    P --> Q[Web Dashboard / Batch Report / Download]
```

### Architectural layers

1. Presentation layer: the Flask UI and browser interface delivered through `templates/` and `static/`.
2. API layer: request handlers in `app.py` that accept image data, dataset requests, and knowledge-base updates.
3. Domain logic layer: `KnowledgeBaseEngine`, inference functions, and symbolic feature classification in `core/engine.py`.
4. Data layer: the in-memory knowledge base and persisted JSON store used for rule management.
5. Vision processing layer: OpenCV-based segmentation and contour processing used to derive visual facts.

## Main Components

### app.py

The Flask application acts as the user interface and API layer.

Responsibilities:

- serve the dashboard at the root route
- expose image analysis endpoints
- provide dataset listing and batch analysis APIs
- manage knowledge-base interactions
- export CSV reports from recognition results

The application loads a `KnowledgeBaseEngine` instance and uses it for all recognition tasks.

### core/engine.py

This file contains the core logic.

Key parts include:

- `KnowledgeBaseEngine`: loads, resets, updates, and saves knowledge rules
- `run_inference`: performs exact and weighted fallback matching
- `analyze_image`: executes the full CV + reasoning pipeline
- feature classifiers for color, shape, and size
- image conversion helpers and similarity scoring functions

## Recognition Pipeline

### 1. Image preprocessing

The engine performs:

- BGR input conversion
- grayscale conversion
- Gaussian blur
- Otsu thresholding
- morphological opening and closing
- contour extraction

This removes noise and helps isolate the foreground object.

### 2. Symbolic feature extraction

Each detected object is mapped to symbolic facts:

- color: red, orange, yellow, green, cyan, blue, purple, pink, white, black, gray
- shape: circle, oval, triangle, square, rectangle, hexagon, or unknown
- size: tiny, small, medium, or large

These are derived from HSV values, contour vertices, aspect ratio, circularity, and bounding-box dimensions.

### 3. Inference and decision-making

The system uses a two-stage inference strategy:

- exact production rule matching: all three symbolic facts match a rule exactly
- weighted fallback matching: if no exact match exists, a fuzzy similarity score is computed

The similarity weights are:

- color: 0.45
- shape: 0.35
- size: 0.20

If the weighted score is above the threshold, the object is recognized as a fallback match. Otherwise it is marked as unknown.

## Knowledge Base

The knowledge base stores objects and associated properties such as:

- name
- color
- shape
- size
- aliases

The default knowledge base is initialized from a set of canonical object rules. A persistent file is used to save changes between sessions:

- `knowledge_base_store.json`

This supports incremental learning and rule edits from the web app.

## API Overview

### Main endpoints

- `GET /` — serves the dashboard
- `GET /api/sample_datasets` — lists dataset image files
- `POST /api/analyze` — analyzes one uploaded or supplied image
- `POST /api/batch_analyze` — analyzes a whole dataset
- `GET /api/knowledge_base` — returns the production rules and current KB state
- `POST /api/add_rule` — adds or updates a rule
- `POST /api/delete_rule` — removes a rule by name
- `POST /api/reset_kb` — restores the default knowledge base
- `POST /api/generate_synthetic` — generates a synthetic image for testing
- `POST /api/export_csv` — exports recognition results to CSV

## Dataset Structure

The project includes several dataset folders:

- `synthetic_object_dataset/` — base synthetic benchmark set
- `knowledge_acquisition_dataset/` — used for incremental learning scenarios
- `unknown_object_dataset/` — used for unknown object testing

The sample index is built dynamically from image files under these folders.

## Launch and Usage

### Install dependencies

```bash
pip install flask flask-cors opencv-python numpy
```

### Run the web app

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

### Regenerate datasets

```bash
python generate_synthetic_dataset.py
python generate_unknown_object_dataset.py
```

## Development Notes

### Adding new rules

Rules may be created through the web interface or programmatically via the `KnowledgeBaseEngine` class. Each rule consists of the following symbolic components:

- color
- shape
- size
- name

The rule is later used in matching and scoring modules.

### Extending the system

Potential extensions include:

- support for more geometric features such as object holes or contour moments
- additional color categories and better HSV calibration
- multi-object detection in a single image
- more robust unknown-class handling
- export features for training and experimentation

## Common Characteristics of Recognized Objects

The project assumes objects are described by a small symbolic vocabulary rather than full image embeddings. This makes the system transparent and explainable, which is beneficial for educational and classical AI workflows.

## Troubleshooting

- If no objects are detected, check image contrast and background separation.
- If recognition results are inconsistent, confirm the HSV thresholding and contour filtering stages are working as expected.
- If a dataset folder is empty, regenerate the dataset or confirm the file path is correct.
- If rules do not persist, verify the project has write access to `knowledge_base_store.json`.

## Summary

This application demonstrates a classical, interpretable AI pipeline: image preprocessing, feature extraction, symbolic reasoning, and production-rule inference. It is a useful example for teaching deterministic computer vision and rule-based decision making without deep-learning models.
