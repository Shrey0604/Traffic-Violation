# Traffic Rule Violation Detection for Two-Wheelers

Detects motorcycles/scooters in a single RGB street image, counts riders per vehicle, flags helmet violations, and extracts the license plate of any violating vehicle — fully offline, with no internet access required at inference time.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
- [Model Choices](#model-choices)
- [Directory Layout](#directory-layout)
- [Output Format](#output-format)
- [Assumptions](#assumptions)
- [Failure Cases & Mitigations](#failure-cases--mitigations)
- [Design Decisions](#design-decisions)
- [Known Limitations](#known-limitations)

---

## Overview

Given a single image, `TrafficViolationDetector.predict()` returns every two-wheeler that is **breaking a rule** — currently:

- **Overcrowding** — more than 2 riders on one vehicle
- **No helmet** — one or more riders without a detected/inferred helmet

For each violating vehicle, the pipeline also attempts to read its license plate via OCR. Compliant vehicles are silently skipped — the output only ever lists violations.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place model weights in ./models/
#    yolov8n.pt          (required  — ~6 MB, COCO-pretrained; auto-downloads via ultralytics if missing)
#    yolov8n.onnx         (optional — ONNX fallback if ultralytics isn't installed)
#    helmet_plate.pt      (optional — custom fine-tuned helmet + plate detector)
#    easyocr/             (required for OCR — EasyOCR English language pack, pre-downloaded for offline use)

# 3. Run
python - <<'EOF'
from solution import TrafficViolationDetector

det = TrafficViolationDetector(model_dir="./models")
result = det.predict("street.jpg")
print(result)
EOF
```

Pre-download the EasyOCR weights once (requires internet that one time):

```python
import easyocr
easyocr.Reader(["en"], model_storage_directory="./models/easyocr")
```

---

## Pipeline

```
Image
  │
  ▼
┌──────────────────────────────────────────────────┐
│ Stage 1 · detect_objects()                        │
│ YOLOv8n → persons, motorcycles                    │
│ + optional aux model → helmets, plates            │
│ (else: heuristic head-region proxy for helmets)   │
└────────────────────┬───────────────────────────────┘
                     │  per detected bike
                     ▼
┌──────────────────────────────────────────────────┐
│ Stage 2 · assign_riders_to_vehicle()              │
│ Expand bike box (riders protrude above frame)     │
│ Person's lower-body centre inside box, OR IoU>0.05│
└────────────────────┬───────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────┐
│ Stage 3 · detect_helmet_violations()              │
│ Head region = top 28% of rider box (±15% width)   │
│ IoU > 0.10 vs. detected helmets, else             │
│ grey std-dev heuristic (σ < 45 → helmet present)  │
└────────────────────┬───────────────────────────────┘
                     ▼
        violation if riders > 2 OR any bare head?
                     │ yes
                     ▼
┌──────────────────────────────────────────────────┐
│ Stage 4 · extract_license_plate()                │
│ Model-detected plate (best IoU vs. lower bike box)│
│ → else Canny + contour search (aspect 1.5–6.0)    │
│ Preprocess: resize → CLAHE → unsharp → adaptive   │
│ threshold → morphological close                   │
│ OCR: EasyOCR (primary) → pytesseract (fallback)   │
└────────────────────┬───────────────────────────────┘
                     ▼
            { "violations": [ {...}, ... ] }
```

Every stage degrades gracefully — a missing model, unreadable image, or failed OCR never raises; it just contributes an empty/default result.

---

## Model Choices

| Model | Size | Role | Why |
|---|---|---|---|
| **YOLOv8n** | ~6 MB (`.pt`) / ~12 MB (`.onnx`) | Primary detector (COCO classes) | Small and fast. Detects `person` (class 0) and `motorcycle` (class 3) out of the box; no custom training needed for the baseline pipeline. |
| **helmet_plate.pt** *(optional)* | ≤ 30 MB | Auxiliary fine-tuned detector | If present, gives direct helmet and license-plate boxes — more accurate than the heuristic fallbacks below. |
| **EasyOCR (en)** | ~40 MB | OCR, primary | Pure-Python, bundles its own weights, no system Tesseract dependency, handles rotated/noisy plate text well. |
| **pytesseract** | system package | OCR, fallback | Lightweight if Tesseract is already installed on the host. |

Worst-case footprint with everything loaded: **~88 MB** — well clear of any tight model-size budget.

When `helmet_plate.pt` is absent, the pipeline falls back to two heuristics instead of failing:
- **Helmets:** standard deviation of grayscale pixel values in the head-crop (a uniform helmet surface has low variance vs. textured hair).
- **License plates:** Canny edge detection + contour search in the lower 40% of the bike's bounding box, filtered by plate-like aspect ratio (1.5–6.0).

---

## Directory Layout

```
.
├── solution.py          # TrafficViolationDetector — the full inference pipeline
├── merge.py             # TODO: document purpose (dataset merge step?)
├── requirements.txt
├── README.md
├── d1/ d2/ d3/          # TODO: document — dataset shards?
├── merged/              # TODO: document — output of merge.py?
└── models/              # not committed — populate locally, see Quick Start
    ├── yolov8n.pt
    ├── helmet_plate.pt  (optional)
    └── easyocr/
```

> Weight files (`yolov8n.pt`, `yolo26n.pt`, etc.) currently live at the repo root rather than under `models/`. Either move them under `models/` to match the loading code in `solution.py`, or update `TrafficViolationDetector(model_dir=...)` calls to point at the repo root. Worth fixing one way or the other so Quick Start works as written for a fresh clone.

---

## Output Format

`predict()` returns a single dict:

```json
{
  "violations": [
    {
      "num_riders": 3,
      "helmet_violations": 1,
      "license_plate": "KA01AB1234"
    }
  ]
}
```

- `num_riders` — number of people assigned to that vehicle.
- `helmet_violations` — count of riders on that vehicle without a detected helmet.
- `license_plate` — OCR'd plate text (alphanumeric only, 2–12 characters), or `""` if no plate could be read.

A vehicle only appears in `violations` if it has more than 2 riders **or** at least one helmet violation. Compliant vehicles are omitted entirely.

---

## Assumptions

1. **Camera angle** — street-level or slightly elevated, such that a motorcycle's bounding box roughly contains its riders.
2. **Rider definition** — anyone whose lower-body centre (70% down their box) falls inside the bike's expanded search region counts as a rider on that vehicle.
3. **Helmet proxy (no aux model)** — top 28% of a rider's box is treated as the head region; low greyscale variance there is read as a helmet.
4. **Vehicle class** — only COCO class 3 (`motorcycle`) is considered; bicycles (class 1) are intentionally excluded.
5. **License plate search** — if no plate is model-detected, the lower 40% of the bike box is scanned for a plate-shaped contour (aspect ratio 1.5–6.0).
6. **Stateless inference** — `predict()` has no memory between calls; each image is processed independently.

---

## Failure Cases & Mitigations

| Failure case | Mitigation |
|---|---|
| No model files in `./models/` | Returns `{"violations": []}` without crashing |
| Image missing or corrupt | `cv2.imread` → `None` → early exit, empty result |
| No two-wheelers detected | Returns `{"violations": []}` immediately |
| Riders partially out of frame | Search box expands 120% upward to catch protruding upper bodies |
| Multiple overlapping bikes | Each bike processed independently via its own search box |
| Plate absent or unreadable | Returns `""` for `license_plate`; never raises |
| OCR returns garbage | `_clean_plate_text` strips non-alphanumeric characters and rejects results outside 2–12 characters |
| Both EasyOCR and pytesseract missing | OCR silently disabled; plate always returned as `""` |
| Unexpected exception anywhere in `predict()` | Caught at the top level, logged, returns `{"violations": []}` |

---

## Design Decisions

**Why per-bike spatial association instead of a global rider count?**
Counting all people in frame and dividing by bike count breaks down in crowded scenes. Instead, each bike gets its own expanded search box, and only people whose lower body falls inside are attributed to it.

**Why IoU 0.10 for helmet matching?**
Helmets (especially with visors) often extend past the modelled head box, so a relaxed threshold avoids penalizing riders who are actually wearing one.

**Why CLAHE + unsharp masking before OCR?**
Plates in street photos are frequently low-contrast and motion-blurred. CLAHE restores local contrast and unsharp masking sharpens character edges before binarization — both measurably help OCR accuracy.

**Why EasyOCR over Tesseract as the primary engine?**
EasyOCR is self-contained (no system Tesseract dependency) and handles rotated or noisy plate text better, which matters more than raw speed here.

**Why a heuristic helmet check at all?**
Without an auxiliary detector, the pipeline still needs to produce an answer. Grey-value standard deviation (σ < 45 → likely helmet) is a cheap, reasonable proxy: helmets are smooth and uniform, hair is high-variance. It's a fallback, not a claim of high accuracy.

---

## Known Limitations

- The helmet heuristic is a coarse proxy, not a trained classifier — expect false positives/negatives without `helmet_plate.pt`.
- Detection thresholds (confidence 0.35, IoU 0.10, etc.) are fixed constants in `solution.py` rather than configurable parameters; tune them in code for your specific camera setup.
- No video/multi-frame tracking — each image is scored independently, so the same vehicle across frames is not deduplicated.

