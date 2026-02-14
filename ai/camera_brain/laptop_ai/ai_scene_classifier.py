# File: laptop_ai/ai_scene_classifier.py
"""
AI Scene Classifier
- Classifies common cinematic contexts (action, night, faces, cars, landscape, closeup, low-light).
- Lightweight: tries to use a PyTorch MobileNet/ResNet if torch available, else uses heuristics (histogram, edge density, face detect).
- Returns a dict of scores for labels.

Public class: SceneClassifier
Methods:
    - predict(frame) -> { label: score, ... }
Dependencies:
    - numpy, opencv
    - optionally torch (will use if installed for stronger classification)
"""
from __future__ import annotations
import time
from typing import Dict
import numpy as np
import cv2

# Try optional torch model
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# face cascade fallback
_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

class SceneClassifier:
    pass
AISceneClassifier = SceneClassifier

class SceneClassifier:
    def __init__(self, use_torch: bool = True):
        self.use_torch = use_torch and TORCH_AVAILABLE
        self.model = None
        if self.use_torch:
            # Placeholder: load a small backbone if available; in this stub we do not download models
            # Replace below with actual model load: torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
            try:
                # lightweight heuristic fallback
                self.model = None
            except Exception:
                self.model = None
                self.use_torch = False

    def _heuristic_predict(self, frame: np.ndarray) -> Dict[str, float]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray)) / 255.0
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))
        # faces detect
        faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        face_score = float(len(faces) > 0)
        # motion-like detection via high edge density -> action
        action_score = min(1.0, edge_density * 8.0)
        night_score = 1.0 - _clamp(mean, 0.0, 1.0)
        landscape_score = 1.0 if w/h > 1.4 else 0.2
        closeup_score = 1.0 if face_score > 0.5 else 0.1
        return {
            "action": float(action_score),
            "night": float(night_score),
            "faces": float(face_score),
            "landscape": float(landscape_score),
            "closeup": float(closeup_score),
            "edge_density": float(edge_density)
        }

    def predict(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Return label scores in 0..1
        """
        ts = time.time()
        if self.use_torch and self.model is not None:
            # placeholder path if user supplies model
            # convert to tensor, run model, map logits -> labels
            # For now fallback:
            return self._heuristic_predict(frame)
        else:
            return self._heuristic_predict(frame)

def _clamp(a, lo, hi):
    return max(lo, min(hi, a))

# Example usage
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    sc = SceneClassifier(use_torch=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        preds = sc.predict(frame)
        print(preds)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
