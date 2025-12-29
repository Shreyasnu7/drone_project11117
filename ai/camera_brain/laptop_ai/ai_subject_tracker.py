# ====================================================================
# File: laptop_ai/ai_subject_tracker.py
# Ultra AI Multi-Subject Tracker
#
# Responsibilities:
#  - Track multiple subjects across frames with stable IDs
#  - Predict future positions (important for curve planning)
#  - Evaluate subject importance (which one should the camera follow?)
#  - Handle occlusions with confidence decay
#  - Fuse YOLO detections + motion + camera metadata
#  - Provide "subject state" to ShotPlanner and CinematicBrain
#
# This is a core module for high-level cinematic intelligence.
# ====================================================================

import numpy as np
import time
from scipy.optimize import linear_sum_assignment


class TrackedSubject:
    """
    Represents one tracked subject (person, vehicle, object).
    Stores:
       - bbox
       - velocity
       - smoothed center
       - class id
       - confidence
       - lost frames count
       - importance score
    """

    def __init__(self, subject_id, bbox, cls, conf):
        self.id = subject_id
        self.bbox = np.array(bbox, dtype=float)
        self.cls = cls
        self.conf = float(conf)

        x1, y1, x2, y2 = bbox
        self.center = np.array([(x1 + x2)/2, (y1 + y2)/2], dtype=float)
        self.velocity = np.zeros(2)

        self.lost_frames = 0
        self.last_update = time.time()
        self.importance = 1.0

    def update(self, bbox, cls, conf):
        """
        Update bbox + velocity + timestamp.
        """
        old_center = self.center.copy()

        self.bbox = np.array(bbox, dtype=float)
        self.cls = cls
        self.conf = float(conf)

        x1, y1, x2, y2 = bbox
        new_center = np.array([(x1 + x2)/2, (y1 + y2)/2], dtype=float)
        self.center = new_center

        dt = time.time() - self.last_update
        if dt > 0:
            self.velocity = (new_center - old_center) / dt

        self.last_update = time.time()
        self.lost_frames = 0

    def predict_position(self, dt=0.3):
        """
        Predicts where the subject will be after dt seconds.
        Cinematic curves use this prediction.
        """
        return self.center + self.velocity * dt

    def distance_to(self, other_center):
        return np.linalg.norm(self.center - other_center)


class AISubjectTracker:
    """
    Core multi-object fusion tracker.
    Think of it as "DeepSORT Lite + Cinematic Awareness".
    """

    def __init__(self, max_lost=8, match_threshold=80):
        self.max_lost = max_lost
        self.match_threshold = match_threshold
        self.subjects = {}  # id -> TrackedSubject
        self.next_id = 1

    # ------------------------------------------------------------------
    # MATCHING LOGIC (Hungarian algorithm)
    # ------------------------------------------------------------------
    def _compute_cost_matrix(self, tracks, detections):
        """
        Cost = Euclidean distance between centers.
        """

        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))

        cost = np.zeros((len(tracks), len(detections)))

        for i, tr in enumerate(tracks):
            for j, det in enumerate(detections):
                x1, y1, x2, y2 = det["bbox"]
                center = np.array([(x1 + x2)/2, (y1 + y2)/2])
                cost[i, j] = np.linalg.norm(tr.center - center)

        return cost

    # ------------------------------------------------------------------
    # MAIN UPDATE
    # ------------------------------------------------------------------
    def update(self, detections):
        """
        detections = [{"bbox":[...], "cls": int, "conf":float}, ...]
        """

        track_list = list(self.subjects.values())
        cost = self._compute_cost_matrix(track_list, detections)

        if len(track_list) > 0 and len(detections) > 0:
            rows, cols = linear_sum_assignment(cost)
        else:
            rows, cols = [], []

        assigned_tracks = set()
        assigned_dets = set()

        # ----------------------------
        # 1. Update matched tracks
        # ----------------------------
        for r, c in zip(rows, cols):
            if cost[r, c] < self.match_threshold:
                track = track_list[r]
                det = detections[c]
                track.update(det["bbox"], det["cls"], det["conf"])
                assigned_tracks.add(r)
                assigned_dets.add(c)

        # ----------------------------
        # 2. Create new tracks
        # ----------------------------
        for i, det in enumerate(detections):
            if i not in assigned_dets:
                sub = TrackedSubject(
                    self.next_id,
                    det["bbox"],
                    det["cls"],
                    det["conf"]
                )
                self.subjects[self.next_id] = sub
                self.next_id += 1

        # ----------------------------
        # 3. Aging & pruning
        # ----------------------------
        remove_list = []
        for i, tr in self.subjects.items():
            if tr.id not in [track_list[r].id for r in assigned_tracks]:
                tr.lost_frames += 1
            if tr.lost_frames > self.max_lost:
                remove_list.append(i)

        for rid in remove_list:
            del self.subjects[rid]

        # ----------------------------
        # Return sorted by importance
        # ----------------------------
        return self.get_ranked_subjects()

    # ------------------------------------------------------------------
    # SUBJECT RANKING FOR CINEMATIC DECISIONS
    # ------------------------------------------------------------------
    def get_ranked_subjects(self):
        """
        Scores subjects based on:
         - confidence
         - motion magnitude
         - centrality in frame
        """

        ranked = []
        for sub in self.subjects.values():
            motion_mag = np.linalg.norm(sub.velocity)
            center_dist = np.linalg.norm(sub.center - np.array([960, 540]))  # image center (placeholder)

            # lower dist => higher importance
            center_score = max(0, 1 - (center_dist / 800))

            score = (
                sub.conf * 0.6 +
                motion_mag * 0.3 +
                center_score * 0.4
            )

            sub.importance = score
            ranked.append(sub)

        ranked.sort(key=lambda x: x.importance, reverse=True)
        return ranked

# ---------------------------
# Advanced features extension
# to laptop_ai/ai_subject_tracker.py
# ---------------------------

import math
from collections import deque, defaultdict
import threading

# ---------------------------
# Simple 2D Kalman Filter (constant velocity)
# ---------------------------
class SimpleKalman2D:
    """
    4-state Kalman filter for [x, y, vx, vy] in image space.
    Very small footprint, good for smoothing and short-term prediction.
    """

    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, dt=1/30.0):
        # State vector [x, y, vx, vy]
        self.dt = float(dt)
        self.x = np.array([x, y, vx, vy], dtype=float)

        # State covariance
        self.P = np.eye(4, dtype=float) * 1.0

        # Process noise (model uncertainty)
        q_pos = 1.0
        q_vel = 5.0
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]) * 0.01

        # Measurement noise (we measure x,y only)
        r_pos = 10.0
        self.R = np.diag([r_pos, r_pos])

        # State transition matrix (constant velocity)
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        # Measurement matrix
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=float)

    def predict(self, dt=None):
        if dt is None:
            dt = self.dt
        # if dt changes we recompute F quickly
        if abs(dt - self.dt) > 1e-6:
            self.dt = dt
            self.F[0, 2] = dt
            self.F[1, 3] = dt

        # x = F x
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return self.x.copy()

    def update(self, meas):
        """
        meas: (x, y)
        """
        z = np.array([meas[0], meas[1]], dtype=float)
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        I = np.eye(4)
        self.P = (I - K.dot(self.H)).dot(self.P)

    def state(self):
        return self.x.copy()

    def pos(self):
        return self.x[0], self.x[1]

    def vel(self):
        return self.x[2], self.x[3]

# ---------------------------
# Appearance embedding & re-ID (lightweight stub)
# ---------------------------
class AppearanceModel:
    """
    Lightweight feature extractor placeholder.
    Replace embed() with a real CNN-based embedding (e.g., MobileNet + global pool).
    For now it calculates simple color-histogram + HOG-lite features for quick similarity.
    """

    def __init__(self, hist_bins=16):
        self.hist_bins = hist_bins

    def embed(self, image_crop):
        # image_crop: HxWx3 (BGR as OpenCV)
        # Convert to HSV and compute normalized color histogram (coarse)
        try:
            hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [self.hist_bins], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [self.hist_bins], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [self.hist_bins], [0, 256])

            # Normalize
            h_hist = (h_hist / (np.sum(h_hist) + 1e-6)).flatten()
            s_hist = (s_hist / (np.sum(s_hist) + 1e-6)).flatten()
            v_hist = (v_hist / (np.sum(v_hist) + 1e-6)).flatten()
            feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)

            # small HOG-like gradient energy (luminance channel)
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            hog_energy = np.array([np.mean(mag), np.std(mag)], dtype=np.float32)

            return np.concatenate([feat, hog_energy])
        except Exception:
            # fallback zero vector
            return np.zeros(self.hist_bins * 3 + 2, dtype=np.float32)

    @staticmethod
    def similarity(a, b):
        """
        Cosine similarity
        """
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

# ---------------------------
# Advanced TrackedSubject (adds kalman + history + appearance)
# ---------------------------
class AdvancedTrackedSubject(TrackedSubject):
    def __init__(self, subject_id, bbox, cls, conf, frame_img=None, max_history=60):
        super().__init__(subject_id, bbox, cls, conf)
        cx, cy = self.center
        self.kf = SimpleKalman2D(x=cx, y=cy, vx=0.0, vy=0.0, dt=1/30.0)
        self.history = deque(maxlen=max_history)  # stores (timestamp, center, bbox, conf)
        self.history.append((self.last_update, self.center.copy(), self.bbox.copy(), self.conf))
        self.appearance = None
        if frame_img is not None:
            self.compute_appearance(frame_img)
        self.reacquire_score = 1.0

    def update(self, bbox, cls, conf, frame_img=None):
        old_center = self.center.copy()
        self.bbox = np.array(bbox, dtype=float)
        x1, y1, x2, y2 = self.bbox
        new_center = np.array([(x1 + x2)/2, (y1 + y2)/2], dtype=float)

        # Kalman predict with dt
        dt = max(1/120.0, time.time() - self.last_update)
        self.kf.predict(dt)
        # update with measurement
        self.kf.update(new_center)
        kstate = self.kf.state()
        self.center = np.array([kstate[0], kstate[1]])
        self.velocity = np.array([kstate[2], kstate[3]])

        self.cls = cls
        self.conf = float(conf)
        self.last_update = time.time()
        self.history.append((self.last_update, self.center.copy(), self.bbox.copy(), self.conf))
        self.lost_frames = 0

        if frame_img is not None:
            self.compute_appearance(frame_img)

    def predict_position(self, dt=0.3):
        # make a copy to avoid changing underlying dt
        self.kf.predict(dt)
        s = self.kf.state()
        return np.array([s[0], s[1]])

    def compute_appearance(self, frame_img):
        """
        Extract appearance embedding from a cropped region of frame_img
        using the current bbox. If cropping fails, skip.
        """
        x1, y1, x2, y2 = map(int, self.bbox)
        h, w = frame_img.shape[:2]
        # clamp
        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame_img[y1:y2, x1:x2].copy()
        am = AppearanceModel()
        self.appearance = am.embed(crop)

    def to_dict(self):
        return {
            "id": self.id,
            "bbox": self.bbox.tolist(),
            "cls": int(self.cls),
            "conf": float(self.conf),
            "center": self.center.tolist(),
            "vel": self.velocity.tolist(),
            "last_update": self.last_update,
            "lost_frames": self.lost_frames,
            "importance": float(self.importance),
        }

# ---------------------------
# Full-fledged AISubjectTracker v2
# ---------------------------
class AdvancedAISubjectTracker(AISubjectTracker):
    """
    A more advanced tracker that uses Kalman smoothing, appearance re-id,
    occlusion handling, and exposes richer APIs for cinematic modules.
    """

    def __init__(self, max_lost=12, match_threshold=150.0, appearance_threshold=0.4):
        super().__init__(max_lost=max_lost, match_threshold=match_threshold)
        # override storage
        self.subjects = {}  # id -> AdvancedTrackedSubject
        self.next_id = 1
        self.appearance_model = AppearanceModel()
        self.appearance_threshold = appearance_threshold
        # For reid fallback: store last embeddings for removed subjects
        self.reid_gallery = {}  # id -> (embedding, last_seen_ts)
        self.lock = threading.Lock()

    def update(self, detections, frame_image=None):
        """
        detections: [{"bbox":[x1,y1,x2,y2], "cls":int, "conf":float}, ...]
        frame_image: full frame image (BGR) for appearance embedding
        """
        with self.lock:
            # convert current subjects to list for cost calc
            track_list = list(self.subjects.values())

            # Build cost matrix using center distance + size difference
            cost = np.zeros((len(track_list), len(detections)), dtype=float)

            for i, tr in enumerate(track_list):
                for j, det in enumerate(detections):
                    x1,y1,x2,y2 = det["bbox"]
                    center = np.array([(x1+x2)/2, (y1+y2)/2])
                    dist = np.linalg.norm(tr.center - center)
                    # size penalty (IoU surrogate)
                    w1 = tr.bbox[2] - tr.bbox[0]
                    h1 = tr.bbox[3] - tr.bbox[1]
                    w2 = x2 - x1
                    h2 = y2 - y1
                    size_pen = abs((w1*h1) - (w2*h2)) / (max(1.0, (w1*h1 + w2*h2)/2.0))
                    cost[i,j] = dist + (size_pen * 50.0)

            if len(track_list) > 0 and len(detections) > 0:
                rows, cols = linear_sum_assignment(cost)
            else:
                rows, cols = [], []

            assigned_tracks = set()
            assigned_dets = set()

            # update matched
            for r, c in zip(rows, cols):
                if cost[r, c] < self.match_threshold:
                    track = track_list[r]
                    det = detections[c]
                    # extract crop for appearance update
                    crop = None
                    if frame_image is not None:
                        x1,y1,x2,y2 = map(int, det["bbox"])
                        h,w = frame_image.shape[:2]
                        x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
                        y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
                        if x2 > x1 and y2 > y1:
                            crop = frame_image[y1:y2, x1:x2].copy()
                    track.update(det["bbox"], det["cls"], det["conf"], frame_img=crop)
                    assigned_tracks.add(track.id)
                    assigned_dets.add(c)

            # create new tracks for unmatched detections
            for i, det in enumerate(detections):
                if i not in assigned_dets:
                    a = AdvancedTrackedSubject(
                        self.next_id, det["bbox"], det["cls"], det["conf"],
                        frame_img=(frame_image if frame_image is not None else None)
                    )
                    self.subjects[self.next_id] = a
                    self.next_id += 1

            # aging & pruning, but keep embeddings for re-id
            remove_list = []
            now = time.time()
            for sid, tr in list(self.subjects.items()):
                if tr.id not in assigned_tracks:
                    tr.lost_frames += 1
                    # degrade confidence slowly
                    tr.conf *= 0.98
                    if tr.lost_frames > self.max_lost:
                        # save last appearance for potential re-id
                        if tr.appearance is not None:
                            self.reid_gallery[tr.id] = (tr.appearance.copy(), now)
                        remove_list.append(sid)

            for rid in remove_list:
                del self.subjects[rid]

            # Attempt re-identification for unmatched detections using appearance
            # only if gallery exists and detection is large enough
            for i, det in enumerate(detections):
                if i in assigned_dets:
                    continue
                if frame_image is None:
                    continue
                x1,y1,x2,y2 = map(int, det["bbox"])
                h,w = frame_image.shape[:2]
                x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
                y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame_image[y1:y2, x1:x2].copy()
                emb = self.appearance_model.embed(crop)
                # scan gallery
                best_id = None
                best_sim = 0.0
                for gid, (gemb, ts) in self.reid_gallery.items():
                    sim = AppearanceModel.similarity(emb, gemb)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = gid
                if best_id is not None and best_sim > self.appearance_threshold:
                    # re-create a subject with the same id
                    new_sub = AdvancedTrackedSubject(best_id, det["bbox"], det["cls"], det["conf"], frame_img=crop)
                    self.subjects[best_id] = new_sub
                    # remove from gallery to avoid dupes
                    if best_id in self.reid_gallery:
                        del self.reid_gallery[best_id]

            # ranking & importance recompute
            ranked = self.get_ranked_subjects()
            return ranked

    def get_subject_state(self, top_k=1):
        """
        Returns structured info about top-k subjects suitable for planner.
        [
           {
             "id": int,
             "bbox": [x1,y1,x2,y2],
             "center": [x,y],
             "vel": [vx,vy],
             "pred_pos": [x,y],
             "class": cls,
             "conf": conf,
             "importance": importance,
             "history": [ ... ] (short list)
           }
        ]
        """
        out = []
        ranked = self.get_ranked_subjects()
        for s in ranked[:top_k]:
            pred = s.predict_position(dt=0.45).tolist()
            hist = list(s.history)[-10:]
            out.append({
                "id": s.id,
                "bbox": [float(x) for x in s.bbox],
                "center": [float(x) for x in s.center],
                "vel": [float(x) for x in s.velocity],
                "pred_pos": [float(x) for x in pred],
                "class": int(s.cls),
                "conf": float(s.conf),
                "importance": float(s.importance),
                "history": [{"ts": h[0], "center": h[1].tolist(), "bbox": h[2].tolist(), "conf": float(h[3])} for h in hist]
            })
        return out

    def export_state(self):
        """
        Persist subject states (ids, last bboxes, embeddings) for crash recovery.
        """
        ser = {"subjects": [], "gallery": {}}
        for sid, s in self.subjects.items():
            ser["subjects"].append(s.to_dict())
            # embeddings are not JSON-serializable: skip or store as list if small
        # save gallery timestamps only
        for gid, (emb, ts) in self.reid_gallery.items():
            ser["gallery"][gid] = {"ts": ts}
        return ser

    def import_state(self, state_json):
        """
        Restore basic IDs (not embeddings) so planners keep continuity after restart.
        """
        # we only restore simple attributes — embedding restoration would need binary store
        for s in state_json.get("subjects", []):
            sid = int(s["id"])
            bbox = s.get("bbox", [0,0,0,0])
            cls = int(s.get("cls", 0))
            conf = float(s.get("conf", 0.1))
            a = AdvancedTrackedSubject(sid, bbox, cls, conf)
            self.subjects[sid] = a
            if sid >= self.next_id:
                self.next_id = sid + 1

    # Debug draw
    def draw_debug(self, frame):
        """
        Draw tracks, velocity and small trail onto frame (mutates).
        """
        for s in self.subjects.values():
            x1,y1,x2,y2 = map(int, s.bbox)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cx, cy = map(int, s.center)
            vx, vy = s.velocity
            cv2.circle(frame, (cx, cy), 3, (0,255,255), -1)
            # velocity arrow
            end = (int(cx + vx*5.0), int(cy + vy*5.0))
            cv2.arrowedLine(frame, (cx, cy), end, (255,0,0), 1, tipLength=0.3)
            cv2.putText(frame, f"id:{s.id} conf:{s.conf:.2f} imp:{s.importance:.2f}",
                        (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            # short trail
            prev_pts = [h[1] for h in list(s.history)[-8:]]
            for i in range(1, len(prev_pts)):
                p0 = tuple(map(int, prev_pts[i-1]))
                p1 = tuple(map(int, prev_pts[i]))
                cv2.line(frame, p0, p1, (0,120,255), 1)
        return frame

# ---------------------------
# Lightweight unit-demo (synthetic frames)
# ---------------------------
if __name__ == "__main__":

    # synthetic test
    tracker = AdvancedAISubjectTracker()

    # create some synthetic detections
    import random
    def create_det(cx, cy, w=80, h=40, cls=2, conf=0.9):
        x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
        return {"bbox":[x1,y1,x2,y2], "cls":cls, "conf":conf}

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    dets = [create_det(900, 520), create_det(1200, 600)]
    ranked = tracker.update(dets, frame_image=frame)
    print("Ranked after 1:", [ (s.id, s.center.tolist()) for s in ranked ])

    # simulate movement
    for t in range(1, 30):
        dets = [create_det(900 + t*3, 520 + t*1.5), create_det(1200 - t*1.2, 600 + math.sin(t/3.0)*4)]
        ranked = tracker.update(dets, frame_image=frame)
        # debug print top subject
        top = tracker.get_subject_state(top_k=1)
        print("t", t, "top", top)
    print("Demo finished.")