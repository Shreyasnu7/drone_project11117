# laptop_ai/vision_tracker.py
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from laptop_ai.config import YOLO_MODEL_PATH, FRAME_SKIP, TEMP_ARTIFACT_DIR
from PIL import Image

os.makedirs(TEMP_ARTIFACT_DIR, exist_ok=True)

class SimpleTrack:
    def __init__(self, tid, bbox, cls, score):
        self.id = tid
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.cls = cls
        self.score = score
        self.hits = 1
        self.last_update = time.time()
        self.velocity = [0.0, 0.0]  # pixels/frame

class SortLikeTracker:
    """
    Very lightweight tracker: nearest-centroid association + simple smoothing.
    Not as robust as DeepSORT, but fast and deterministic.
    """
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracks = []
        self._next_id = 1

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw * ih
        areaA = max(0, ax2-ax1)*max(0, ay2-ay1)
        areaB = max(0, bx2-bx1)*max(0, by2-by1)
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    def update(self, detections):
        """
        detections: list of [x1,y1,x2,y2, cls, score]
        """
        now = time.time()
        assigned = [False]*len(detections)
        # compute IOU matrix
        for tr in self.tracks:
            best_iou = 0.0
            best_idx = -1
            for i, det in enumerate(detections):
                if assigned[i]: continue
                iou_val = self.iou(tr.bbox, det[:4])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            if best_idx >= 0 and best_iou >= self.iou_threshold:
                det = detections[best_idx]
                # update velocity
                cx_old = (tr.bbox[0] + tr.bbox[2])/2.0
                cy_old = (tr.bbox[1] + tr.bbox[3])/2.0
                cx_new = (det[0] + det[2])/2.0
                cy_new = (det[1] + det[3])/2.0
                dt = max(1e-3, now - tr.last_update)
                tr.velocity = [(cx_new-cx_old)/dt, (cy_new-cy_old)/dt]
                tr.bbox = det[:4]
                tr.score = det[5]
                tr.hits += 1
                tr.last_update = now
                assigned[best_idx] = True
            else:
                # aged track remains; do not delete immediately
                pass

        # add unassigned detections as new tracks
        for i, det in enumerate(detections):
            if not assigned[i]:
                tr = SimpleTrack(self._next_id, det[:4], det[4], det[5])
                self._next_id += 1
                self.tracks.append(tr)

        # remove old tracks
        self.tracks = [t for t in self.tracks if (now - t.last_update) <= self.max_age]

        # return list of dicts
        out = []
        for t in self.tracks:
            out.append({
                "id": t.id,
                "bbox": [float(x) for x in t.bbox],
                "cls": int(t.cls),
                "score": float(t.score),
                "hits": int(t.hits),
                "last_update": t.last_update,
                "velocity": [float(t.velocity[0]), float(t.velocity[1])]
            })
        return out

class VisionTracker:
    def __init__(self, model_path=None, target_classes=None):
        self.model_path = model_path or YOLO_MODEL_PATH
        print("Loading YOLO model (this may take a sec)...")
        self.model = YOLO(self.model_path)  # relies on ultralytics to use GPU if torch.cuda available
        self.frame_count = 0
        self.tracker = SortLikeTracker(max_age=30, iou_threshold=0.3)
        self.target_classes = target_classes if target_classes is not None else [0,1,2]  # person,bike,car default
        self._last_annot_save = 0

    def _dets_from_result(self, r):
        dets = []
        # r.boxes available format from ultralytics
        for box in r.boxes:
            try:
                cls = int(box.cls.cpu().numpy()[0])
                score = float(box.conf.cpu().numpy()[0])
                if cls not in self.target_classes:
                    continue
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().tolist()
                dets.append([float(x1), float(y1), float(x2), float(y2), cls, score])
            except Exception:
                continue
        return dets

    def process_frame(self, frame):
        """
        Input: BGR numpy image
        Returns: vision_context, annotated_image (BGR)
        """
        self.frame_count += 1

        # Run YOLO every FRAME_SKIP frames
        if (self.frame_count % (FRAME_SKIP or 1)) != 0:
            # just return empty/no-op context
            return {"tracks":[], "selected": None, "frame_id": self.frame_count, "timestamp": time.time()}, frame

        # ultralytics returns list of results
        res = self.model.track(frame, persist=True, verbose=False)[0]
        dets = self._dets_from_result(res)
        tracks = self.tracker.update(dets)

        # choose best track (highest hits, freshest)
        selected = None
        best_score = -1
        now = time.time()
        for t in tracks:
            score = t["hits"] - (now - t["last_update"])*0.1
            if score > best_score:
                best_score = score
                selected = t

        annotated = frame.copy()
        for t in tracks:
            x1,y1,x2,y2 = map(int, t["bbox"])
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (16,200,16), 2)
            cv2.putText(annotated, f"id:{t['id']} cls:{t['cls']} vx:{int(t['velocity'][0])}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # occasionally save a debug frame
        if time.time() - self._last_annot_save > 8.0:
            try:
                fname = f"{TEMP_ARTIFACT_DIR}/annot_{int(time.time())}.jpg"
                Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(fname)
                self._last_annot_save = time.time()
            except Exception:
                pass

        vision_context = {
            "tracks": tracks,
            "selected": selected,
            "frame_id": self.frame_count,
            "timestamp": time.time()
        }
        return vision_context, annotated
