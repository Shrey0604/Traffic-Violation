"""
Traffic Rule Violation Detection for Two-Wheelers
==================================================
Detects motorcycles/scooters, counts riders, identifies helmet violations,
and extracts license plate text for violating vehicles.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: IoU / spatial utilities
# ---------------------------------------------------------------------------

def _iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    areaA = max(1e-6, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1e-6, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return interArea / (areaA + areaB - interArea + 1e-6)


def _box_center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _box_area(box: List[float]) -> float:
    return max(0.0, (box[2] - box[0]) * (box[3] - box[1]))


def _expand_box(box: List[float], factor: float, img_w: int, img_h: int) -> List[float]:
    """Expand a box by `factor` while keeping it within image bounds."""
    cx, cy = _box_center(box)
    w = (box[2] - box[0]) * factor
    h = (box[3] - box[1]) * factor
    return [
        max(0, cx - w / 2),
        max(0, cy - h / 2),
        min(img_w, cx + w / 2),
        min(img_h, cy + h / 2),
    ]


# ---------------------------------------------------------------------------
# Main Detector Class
# ---------------------------------------------------------------------------

class TrafficViolationDetector:
    """
    Detects traffic rule violations on two-wheelers from a single RGB image.

    Pipeline
    --------
    1. Load YOLOv8n (or YOLOv5/ONNX fallback) once in __init__.
    2. predict() runs the full pipeline:
       a. detect_objects()          – YOLO inference
       b. assign_riders_to_vehicle()– spatial association
       c. detect_helmet_violations()– helmet ↔ rider matching
       d. extract_license_plate()   – OCR with EasyOCR or pytesseract
    """

    # COCO class ids (YOLOv8 defaults)
    _CLS_PERSON       = 0
    _CLS_MOTORCYCLE   = 3   # bicycle=1, motorcycle=3, bus=5, car=2, truck=7
    # custom / fine-tuned heads we fall back on if present
    _CLS_HELMET_IDS   = {0}   # overridden after model load if custom
    _CLS_PLATE_IDS    = {0}   # overridden after model load if custom

    def __init__(self, model_dir: str = "./models"):
        """
        Load all models from `model_dir`.  No internet access occurs here.

        Expected files (at least one detector must be present):
          ./models/yolov8n.pt          – preferred (ultralytics, ~6 MB)
          ./models/yolov8n.onnx        – ONNX fallback (~12 MB)
          ./models/helmet_plate.pt     – optional fine-tuned head detector
        """
        self.model_dir = Path(model_dir)
        self._detector = None          # primary two-wheeler / person detector
        self._aux_detector = None      # optional helmet + plate detector
        self._ocr_reader = None
        self._use_easyocr = False
        self._img_w = 0
        self._img_h = 0

        self._load_detector()
        self._load_ocr()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_detector(self) -> None:
        """Load YOLO detector – tries ultralytics → ONNX → OpenCV DNN."""
        pt_path   = self.model_dir / "yolov8n.pt"
        onnx_path = self.model_dir / "yolov8n.onnx"
        aux_path  = self.model_dir / "helmet_plate.pt"

        # 1) Try ultralytics YOLOv8
        if pt_path.exists():
            try:
                from ultralytics import YOLO
                self._detector = YOLO(str(pt_path))
                self._detector_backend = "ultralytics"
                logger.info("Loaded YOLOv8 (ultralytics) from %s", pt_path)

                # Optional auxiliary detector for helmets & plates
                if aux_path.exists():
                    self._aux_detector = YOLO(str(aux_path))
                    logger.info("Loaded auxiliary detector from %s", aux_path)
                return
            except Exception as e:
                logger.warning("ultralytics load failed: %s", e)

        # 2) ONNX via OpenCV DNN
        if onnx_path.exists():
            try:
                net = cv2.dnn.readNetFromONNX(str(onnx_path))
                self._detector = net
                self._detector_backend = "onnx"
                logger.info("Loaded YOLOv8 ONNX from %s", onnx_path)
                return
            except Exception as e:
                logger.warning("ONNX load failed: %s", e)

        # 3) No model found – use a heuristic-only stub (returns no detections)
        logger.warning(
            "No detection model found in %s. "
            "Detector will return empty results.", self.model_dir
        )
        self._detector = None
        self._detector_backend = "none"

    def _load_ocr(self) -> None:
        """Load EasyOCR (preferred) or fall back to pytesseract."""
        try:
            import easyocr
            # gpu=False keeps it lightweight; model files must be pre-downloaded
            self._ocr_reader = easyocr.Reader(
                ["en"],
                gpu=False,
                model_storage_directory=str(self.model_dir / "easyocr"),
                download_enabled=False,
            )
            self._use_easyocr = True
            logger.info("EasyOCR loaded.")
        except Exception as e:
            logger.warning("EasyOCR unavailable (%s), will try pytesseract.", e)
            try:
                import pytesseract
                self._ocr_reader = pytesseract
                self._use_easyocr = False
                logger.info("pytesseract loaded.")
            except Exception as e2:
                logger.warning("pytesseract also unavailable (%s). OCR disabled.", e2)
                self._ocr_reader = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> dict:
        """
        Run full violation detection pipeline on a single image.

        Parameters
        ----------
        image_path : str
            Absolute or relative path to an RGB image file.

        Returns
        -------
        dict
            {"violations": [...]} or {"violations": []}
        """
        try:
            image = self._load_image(image_path)
            if image is None:
                return {"violations": []}

            self._img_h, self._img_w = image.shape[:2]

            # Stage 1 – Object detection
            detections = self.detect_objects(image)

            bikes    = detections.get("bikes", [])
            persons  = detections.get("persons", [])
            helmets  = detections.get("helmets", [])
            plates   = detections.get("plates", [])

            if not bikes:
                return {"violations": []}

            violations = []

            for bike_box in bikes:
                # Stage 2 – Associate riders with this vehicle
                riders = self.assign_riders_to_vehicle(bike_box, persons, image)

                num_riders = len(riders)

                # Stage 3 – Helmet violation counting
                helmet_violations = self.detect_helmet_violations(
                    riders, helmets, image
                )

                # Determine if this vehicle is violating
                overfull = num_riders > 2
                has_helmet_violation = helmet_violations > 0

                if not (overfull or has_helmet_violation):
                    continue  # compliant vehicle – skip

                # Stage 4 – License plate extraction
                plate_text = self.extract_license_plate(
                    bike_box, plates, image
                )

                violations.append(
                    {
                        "num_riders": num_riders,
                        "helmet_violations": helmet_violations,
                        "license_plate": plate_text,
                    }
                )

            return {"violations": violations}

        except Exception as e:
            logger.error("predict() raised an unexpected error: %s", e, exc_info=True)
            return {"violations": []}

    # ------------------------------------------------------------------
    # Stage 1 – Object detection
    # ------------------------------------------------------------------

    def detect_objects(self, image: np.ndarray) -> Dict[str, List[List[float]]]:
        """
        Run YOLO inference and return categorised bounding boxes.

        Returns
        -------
        dict with keys: 'bikes', 'persons', 'helmets', 'plates'
        Each value is a list of [x1, y1, x2, y2, conf] lists.
        """
        result = {"bikes": [], "persons": [], "helmets": [], "plates": []}

        if self._detector is None:
            return result

        try:
            if self._detector_backend == "ultralytics":
                result = self._detect_ultralytics(image, result)
            elif self._detector_backend == "onnx":
                result = self._detect_onnx(image, result)
        except Exception as e:
            logger.error("detect_objects() error: %s", e, exc_info=True)

        return result

    def _detect_ultralytics(
        self, image: np.ndarray, result: dict
    ) -> dict:
        """Run ultralytics YOLO and fill result dict."""
        # Primary COCO model
        preds = self._detector(image, verbose=False)[0]
        for box in preds.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            entry = [x1, y1, x2, y2, conf]

            if cls == self._CLS_PERSON and conf > 0.35:
                result["persons"].append(entry)
            elif cls == self._CLS_MOTORCYCLE and conf > 0.35:
                result["bikes"].append(entry)

        # Auxiliary helmet / plate model (custom fine-tuned)
        if self._aux_detector is not None:
            aux_preds = self._aux_detector(image, verbose=False)[0]
            cls_names = aux_preds.names  # {id: name}
            for box in aux_preds.boxes:
                cls   = int(box.cls[0])
                conf  = float(box.conf[0])
                name  = cls_names.get(cls, "").lower()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                entry = [x1, y1, x2, y2, conf]
                if "helmet" in name and conf > 0.30:
                    result["helmets"].append(entry)
                elif "plate" in name and conf > 0.25:
                    result["plates"].append(entry)
                elif "no_helmet" in name or "nohelmet" in name:
                    # auxiliary model directly labels helmetless riders
                    pass
        else:
            # Heuristic helmet proxy: upper-body region of each person
            result["helmets"] = self._heuristic_helmet_regions(
                image, result["persons"]
            )

        return result

    def _detect_onnx(self, image: np.ndarray, result: dict) -> dict:
        """
        Run YOLOv8 ONNX via OpenCV DNN.
        Input size: 640×640, outputs shape [1, 84, 8400].
        """
        INPUT_SIZE = 640
        CONF_THRESH = 0.35
        NMS_THRESH  = 0.45

        img_h, img_w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
            swapRB=True, crop=False
        )
        self._detector.setInput(blob)
        raw = self._detector.forward()          # [1, 84, 8400]
        raw = raw[0].T                          # [8400, 84]

        scale_x = img_w / INPUT_SIZE
        scale_y = img_h / INPUT_SIZE

        boxes_out, confs_out, cls_out = [], [], []
        for row in raw:
            cx, cy, w, h = row[:4]
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf   = float(class_scores[cls_id])
            if conf < CONF_THRESH:
                continue
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            x2 = (cx + w / 2) * scale_x
            y2 = (cy + h / 2) * scale_y
            boxes_out.append([x1, y1, x2 - x1, y2 - y1])
            confs_out.append(conf)
            cls_out.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes_out, confs_out, CONF_THRESH, NMS_THRESH)
        if len(indices) == 0:
            return result

        for i in indices.flatten():
            x, y, w, h = boxes_out[i]
            x1, y1, x2, y2 = x, y, x + w, y + h
            conf  = confs_out[i]
            cls   = cls_out[i]
            entry = [x1, y1, x2, y2, conf]
            if cls == self._CLS_PERSON:
                result["persons"].append(entry)
            elif cls == self._CLS_MOTORCYCLE:
                result["bikes"].append(entry)

        result["helmets"] = self._heuristic_helmet_regions(
            image, result["persons"]
        )
        return result

    # ------------------------------------------------------------------
    # Stage 2 – Rider-to-vehicle association
    # ------------------------------------------------------------------

    def assign_riders_to_vehicle(
        self,
        bike_box: List[float],
        persons: List[List[float]],
        image: np.ndarray,
    ) -> List[List[float]]:
        """
        Assign person detections to a given vehicle bounding box.

        Strategy
        --------
        * Expand the bike box slightly (riders can protrude upward).
        * Accept a person if:
          - Their lower half overlaps the expanded bike region (IoU > 0 or
            centre-of-lower-body inside expanded box), OR
          - Their centre is within 1× the bike width horizontally and
            their bottom edge is near the bike top edge.

        Returns list of person boxes assigned to this bike.
        """
        img_h, img_w = image.shape[:2]
        bx1, by1, bx2, by2 = bike_box[:4]
        bw = bx2 - bx1
        bh = by2 - by1

        # Expand bike box upward (riders sit above the frame) and slightly sideways
        search_x1 = max(0,     bx1 - bw * 0.15)
        search_y1 = max(0,     by1 - bh * 1.20)   # riders protrude above
        search_x2 = min(img_w, bx2 + bw * 0.15)
        search_y2 = min(img_h, by2 + bh * 0.05)
        search_box = [search_x1, search_y1, search_x2, search_y2]

        assigned = []
        for person in persons:
            px1, py1, px2, py2 = person[:4]

            # Lower-body centre of person
            lower_cx = (px1 + px2) / 2.0
            lower_cy = py1 + (py2 - py1) * 0.70  # 70% down = hip/leg region

            in_search = (
                search_x1 <= lower_cx <= search_x2
                and search_y1 <= lower_cy <= search_y2
            )

            overlap = _iou([px1, py1, px2, py2], search_box) > 0.05

            if in_search or overlap:
                assigned.append(person)

        return assigned

    # ------------------------------------------------------------------
    # Stage 3 – Helmet violation detection
    # ------------------------------------------------------------------

    def detect_helmet_violations(
        self,
        riders: List[List[float]],
        helmets: List[List[float]],
        image: np.ndarray,
    ) -> int:
        """
        Count riders who are NOT wearing a helmet.

        Algorithm
        ---------
        For each rider, define a "head region" as the top-25% of their
        bounding box (expanded by 30% width).  Check if any detected
        helmet overlaps that head region with IoU > threshold OR if a
        heuristic colour/edge check suggests a helmet is present.

        Returns the number of helmet violations.
        """
        violations = 0
        IOU_THRESH = 0.10  # relaxed: partial overlap counts as wearing

        for rider in riders:
            rx1, ry1, rx2, ry2 = rider[:4]
            rw = rx2 - rx1
            rh = ry2 - ry1

            # Head region: top ~25% of rider box, slightly expanded
            head_box = [
                max(0, rx1 - rw * 0.15),
                ry1,
                min(self._img_w, rx2 + rw * 0.15),
                ry1 + rh * 0.28,
            ]

            helmet_found = False
            for helmet in helmets:
                hx1, hy1, hx2, hy2 = helmet[:4]
                iou = _iou(head_box, [hx1, hy1, hx2, hy2])
                if iou > IOU_THRESH:
                    helmet_found = True
                    break

            if not helmet_found:
                # Heuristic fallback: colour-based helmet detection in head crop
                helmet_found = self._heuristic_helmet_check(image, head_box)

            if not helmet_found:
                violations += 1

        return violations

    # ------------------------------------------------------------------
    # Stage 4 – License plate extraction
    # ------------------------------------------------------------------

    def extract_license_plate(
        self,
        bike_box: List[float],
        plates: List[List[float]],
        image: np.ndarray,
    ) -> str:
        """
        Extract and OCR the license plate text for a given vehicle.

        Strategy
        --------
        1. Use detected plate boxes (if any) that spatially belong to this bike.
        2. Fall back to scanning the lower portion of the bike bounding box
           using edge/contour analysis to find a plate-shaped region.
        3. Preprocess crop: grayscale → CLAHE → sharpening → adaptive thresh.
        4. Run OCR (EasyOCR / pytesseract).

        Returns cleaned plate string, or "" if not found.
        """
        try:
            plate_crop = self._find_plate_crop(bike_box, plates, image)
            if plate_crop is None or plate_crop.size == 0:
                return ""

            processed = self._preprocess_plate(plate_crop)
            text = self._run_ocr(processed)
            return self._clean_plate_text(text)

        except Exception as e:
            logger.warning("extract_license_plate() error: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from disk, return BGR numpy array or None."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error("Could not read image: %s", image_path)
                return None
            return img
        except Exception as e:
            logger.error("Image load error: %s", e)
            return None

    def _heuristic_helmet_regions(
        self, image: np.ndarray, persons: List[List[float]]
    ) -> List[List[float]]:
        """
        When no auxiliary helmet detector is available, generate candidate
        helmet boxes as the top-25% of each person bounding box.
        These are later matched against colour/texture heuristics.
        """
        helmets = []
        for p in persons:
            px1, py1, px2, py2 = p[:4]
            rh = py2 - py1
            helmets.append([px1, py1, px2, py1 + rh * 0.25, 0.5])
        return helmets

    def _heuristic_helmet_check(
        self, image: np.ndarray, head_box: List[float]
    ) -> bool:
        """
        Simple colour/shape heuristic: helmets tend to be large, rounded,
        single-colour blobs (often dark).  Not a replacement for a proper model
        but reduces false positives in heuristic-only mode.
        """
        try:
            x1, y1, x2, y2 = [int(v) for v in head_box]
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return False

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Standard deviation: a helmet is a coherent coloured region
            std = float(np.std(gray))
            # High variance → textured hair (no helmet); low variance → solid helmet
            return std < 45.0

        except Exception:
            return False

    def _find_plate_crop(
        self,
        bike_box: List[float],
        plates: List[List[float]],
        image: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Locate the license plate crop for a given bike.
        Preference order: model-detected plate → contour heuristic.
        """
        bx1, by1, bx2, by2 = bike_box[:4]

        # 1) Use model-detected plates that overlap with the lower portion of this bike
        lower_bike = [bx1, (by1 + by2) / 2.0, bx2, by2]
        best_iou   = 0.0
        best_plate = None
        for plate in plates:
            px1, py1, px2, py2 = plate[:4]
            iou = _iou(lower_bike, [px1, py1, px2, py2])
            if iou > best_iou:
                best_iou   = iou
                best_plate = plate

        if best_plate is not None and best_iou > 0.02:
            px1, py1, px2, py2 = [int(v) for v in best_plate[:4]]
            return image[py1:py2, px1:px2]

        # 2) Contour-based fallback: search lower 40% of bike box
        try:
            x1, y1, x2, y2 = int(bx1), int(by1), int(bx2), int(by2)
            bh = y2 - y1
            roi_y1 = y1 + int(bh * 0.55)
            roi = image[roi_y1:y2, x1:x2]
            if roi.size == 0:
                return None

            gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur  = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            cnts, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            candidates = []
            for cnt in cnts:
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                aspect = rw / max(rh, 1)
                area   = rw * rh
                if 1.5 < aspect < 6.0 and area > 400:
                    candidates.append((area, rx, ry, rw, rh))

            if not candidates:
                return None

            candidates.sort(key=lambda c: -c[0])
            _, rx, ry, rw, rh = candidates[0]

            # Add small padding
            pad = 4
            crop_y1 = max(0, ry - pad)
            crop_y2 = min(roi.shape[0], ry + rh + pad)
            crop_x1 = max(0, rx - pad)
            crop_x2 = min(roi.shape[1], rx + rw + pad)
            return roi[crop_y1:crop_y2, crop_x1:crop_x2]

        except Exception as e:
            logger.warning("Plate contour search failed: %s", e)
            return None

    def _preprocess_plate(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for OCR:
        grayscale → resize → CLAHE → unsharp mask → adaptive threshold.
        """
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Resize to at least 80px tall for OCR
        h, w = gray.shape
        target_h = max(80, h)
        scale    = target_h / h
        resized  = cv2.resize(
            gray, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC
        )

        # CLAHE contrast enhancement
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        equated = clahe.apply(resized)

        # Unsharp mask
        blurred  = cv2.GaussianBlur(equated, (0, 0), 3)
        sharp    = cv2.addWeighted(equated, 1.8, blurred, -0.8, 0)

        # Adaptive threshold (binarise)
        binary = cv2.adaptiveThreshold(
            sharp, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 8
        )

        # Morphological cleanup
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _run_ocr(self, processed: np.ndarray) -> str:
        """Run OCR on preprocessed plate image."""
        if self._ocr_reader is None:
            return ""

        try:
            if self._use_easyocr:
                results = self._ocr_reader.readtext(
                    processed,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                    detail=0,
                    paragraph=False,
                )
                return " ".join(results)
            else:
                # pytesseract
                import pytesseract
                config = (
                    "--psm 8 --oem 1 "
                    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )
                return pytesseract.image_to_string(processed, config=config)
        except Exception as e:
            logger.warning("OCR error: %s", e)
            return ""

    @staticmethod
    def _clean_plate_text(raw: str) -> str:
        """Strip whitespace and non-alphanumeric chars from OCR output."""
        import re
        cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
        # Reject obviously bad reads (too short or too long)
        if len(cleaned) < 2 or len(cleaned) > 12:
            return ""
        return cleaned
