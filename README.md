# Traffic Rule Violation Detection for Two-Wheelers

Detects motorcycles/scooters in a single RGB street image, counts riders per vehicle, identifies helmet violations, and extracts the license plate of any violating vehicle — all offline, with no internet access at runtime.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place model weights
#    ./models/yolov8n.pt          (required — ~6 MB, COCO-pretrained)
#    ./models/helmet_plate.pt     (optional — fine-tuned auxiliary detector)
#    ./models/easyocr/            (required — EasyOCR language pack, pre-downloaded)

# 3. Run
python - <<'EOF'
from solution import TrafficViolationDetector
det = TrafficViolationDetector(model_dir="./models")
print(det.predict("street.jpg"))
EOF
```

---

## Pipeline Explanation

```
Image
  │
  ▼
┌─────────────────────────────────────────────────┐
│  Stage 1 · detect_objects()                     │
│  YOLOv8n → bikes, persons, helmets, plates      │
└───────────────────┬─────────────────────────────┘
                    │  per-bike loop
                    ▼
┌─────────────────────────────────────────────────┐
│  Stage 2 · assign_riders_to_vehicle()           │
│  Spatial overlap: expanded bike search box      │
│  lower-body centre check + IoU filter           │
└───────────────────┬─────────────────────────────┘
                    ▼
┌─────────────────────────────────────────────────┐
│  Stage 3 · detect_helmet_violations()           │
│  Head region (top 25% of rider box)             │
│  IoU match against helmet detections            │
│  Colour heuristic fallback (std-dev of grey)    │
└───────────────────┬─────────────────────────────┘
                    ▼
          Violation? (>2 riders OR helmet viol.)
                    │ yes
                    ▼
┌─────────────────────────────────────────────────┐
│  Stage 4 · extract_license_plate()             │
│  Model-detected plate → contour fallback        │
│  Preprocessing: CLAHE → unsharp → adaptive thr  │
│  OCR: EasyOCR (primary) / pytesseract (fallback)│
└─────────────────────────────────────────────────┘
                    │
                    ▼
            {"violations": [...]}
```

---

## Model Choices

| Model | Size | Role | Why |
|---|---|---|---|
| **YOLOv8n** | ~6 MB (.pt) / ~12 MB (.onnx) | Primary detector (COCO) | Fast, accurate, well under 250 MB cap. Detects `person` (cls 0) and `motorcycle` (cls 3) out of the box. |
| **helmet_plate.pt** *(optional)* | ≤ 30 MB | Auxiliary fine-tuned detector | Provides direct helmet & license-plate boxes for better accuracy when available. |
| **EasyOCR** (en) | ~40 MB | OCR | Offline, no Tesseract system dependency, good accuracy on licence plates. |
| **pytesseract** | system | OCR fallback | Extremely lightweight if Tesseract is already installed. |

Total worst-case footprint: **~88 MB** (well under 250 MB).

---

## Directory Layout

```
./
├── solution.py
├── requirements.txt
├── README.md
└── models/
    ├── yolov8n.pt          ← download once: ultralytics hub or official repo
    ├── helmet_plate.pt     ← optional custom model
    └── easyocr/
        └── craft_mlt_25k.pth, english_g2.pth ...
```

To pre-download EasyOCR models (do this once, with internet):

```python
import easyocr
easyocr.Reader(["en"], model_storage_directory="./models/easyocr")
```

---

## Assumptions

1. **Camera angle**: The detector assumes a street-level or slight overhead perspective where motorcycle bounding boxes contain (or nearly contain) the riders.
2. **Rider definition**: Anyone whose lower-body centre falls inside the expanded vehicle search region is counted as a rider on that vehicle.
3. **Helmet proxy** (no auxiliary model): The top 25% of a rider bounding box is treated as the head region. A low grey standard deviation in that crop is interpreted as a solid-coloured helmet.
4. **Two-wheeler classes**: Only COCO class 3 (`motorcycle`) is considered. Bicycles (class 1) are excluded because the task specifies motorised scooters/motorcycles.
5. **License plate**: If no plate is detected by the model, the lower 40% of the bike box is searched via edge/contour analysis for a rectangle with aspect ratio 1.5–6.0.
6. **One image**: The `predict()` method is fully stateless — each call is independent.

---

## Failure Cases & Mitigations

| Failure case | Mitigation |
|---|---|
| No model files in `./models/` | Gracefully returns `{"violations": []}` without crashing |
| Image file missing or corrupt | `cv2.imread` returns None → early exit with empty violations |
| No two-wheelers detected | Returns `{"violations": []}` immediately |
| Riders partially out of frame | Expanded search box (120% height upward) captures partially visible riders |
| Multiple overlapping bikes | Per-bike loop with spatial association ensures each vehicle processed independently |
| License plate absent / unreadable | Returns `""` for `license_plate`; never raises exception |
| OCR produces garbage | `_clean_plate_text` strips non-alphanumeric chars and rejects reads shorter than 2 or longer than 12 chars |
| EasyOCR & pytesseract both missing | OCR silently disabled; plate returned as `""` |

---

## Design Decisions

### Why per-bike spatial association instead of global counting?
Counting all persons in the image and dividing by number of bikes would be wildly inaccurate in crowded scenes. Instead, each bike gets its own **expanded search box** (extends 120% above the frame to capture riders whose upper bodies protrude) and only persons whose lower-body centre falls inside are assigned.

### Why IoU 0.10 for helmet matching?
Helmets often extend slightly beyond the modelled head bounding box due to visors, so a relaxed threshold avoids false violations.

### Why CLAHE + unsharp mask for plate preprocessing?
License plates in street images are frequently low-contrast and motion-blurred. CLAHE restores local contrast; unsharp masking sharpens character edges before binarisation, both of which significantly improve OCR accuracy.

### Why EasyOCR over Tesseract as primary?
EasyOCR is a self-contained Python library that bundles its own DNN models. It does not require a system-level Tesseract installation and performs better on rotated / noisy text — common in real-world plate images.

### Why the heuristic helmet check?
When no auxiliary helmet detector model is provided, the solution must still operate. The grey standard-deviation heuristic (σ < 45 → likely helmet) is a reasonable proxy: helmets present a smooth, uniform surface while hair exhibits high texture variance. It degrades gracefully rather than crashing.
