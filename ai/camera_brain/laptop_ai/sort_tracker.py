# File: laptop_ai/sort_tracker.py
# Minimal SORT-style tracker using a 2D constant-velocity Kalman filter for bbox centers.
# Keeps dependency lightweight (uses filterpy).

import numpy as np
from filterpy.kalman import KalmanFilter
import time
import math

class Track:
    def __init__(self, bbox, track_id):
        # bbox = [x1,y1,x2,y2]
        self.bbox = np.array(bbox, dtype=float)
        self.id = track_id
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        dt = 1.0
        # state: [cx, cy, vx, vy, w, h]
        self.kf.F = np.array([
            [1,0,dt,0,0,0],
            [0,1,0,dt,0,0],
            [0,0,1,0,0,0],
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,0,0,1,0]
        ])
        self.kf.P *= 10.0
        self.kf.R *= 1.0
        cx = (bbox[0]+bbox[2])/2.0
        cy = (bbox[1]+bbox[3])/2.0
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        self.kf.x = np.array([cx, cy, 0, 0, w, h]).reshape((6,1))
        self.time_since_update = 0
        self.age = 0
        self.hits = 1
        self.last_update_time = time.time()

    def predict(self, dt=1.0):
        # predict step
        self.kf.F[0,2] = dt
        self.kf.F[1,3] = dt
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox):
        cx = (bbox[0]+bbox[2])/2.0
        cy = (bbox[1]+bbox[3])/2.0
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        z = np.array([cx, cy, w]).reshape((3,1))
        self.kf.update(z)
        self.hits += 1
        self.time_since_update = 0
        self.last_update_time = time.time()
        # reconstruct bbox from state
        cx, cy, _, _, w, h = self.kf.x.flatten()
        x1 = cx - w/2.0
        y1 = cy - h/2.0
        x2 = cx + w/2.0
        y2 = cy + h/2.0
        self.bbox = np.array([x1,y1,x2,y2])

    def as_dict(self):
        cx = float(self.kf.x[0])
        cy = float(self.kf.x[1])
        vx = float(self.kf.x[2])
        vy = float(self.kf.x[3])
        w = float(self.kf.x[4])
        h = float(self.kf.x[5])
        return {
            "id": self.id,
            "bbox": [float(self.bbox[0]), float(self.bbox[1]), float(self.bbox[2]), float(self.bbox[3])],
            "center": [cx, cy],
            "velocity": [vx, vy],
            "size": [w, h],
            "age": self.age,
            "hits": self.hits,
            "last_update": self.last_update_time
        }

class SortTracker:
    def __init__(self, max_age=30, iou_threshold=0.3):
        self.tracks = []
        self.next_id = 1
        self.max_age = max_age
        self.iou_threshold = iou_threshold

    @staticmethod
    def iou(bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2-xx1)
        h = np.maximum(0., yy2-yy1)
        inter = w*h
        area1 = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        area2 = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
        o = inter / (area1 + area2 - inter + 1e-6)
        return o

    def update(self, detections):
        # detections: list of [x1,y1,x2,y2]
        # predict all
        for t in self.tracks:
            t.predict()
        matches, unmatched_dets, unmatched_trks = self._associate(detections)

        # update matched
        for tidx, didx in matches:
            self.tracks[tidx].update(detections[didx])

        # create new tracks for unmatched detections
        for didx in unmatched_dets:
            tr = Track(detections[didx], self.next_id)
            self.next_id += 1
            self.tracks.append(tr)

        # remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # return active tracks as dicts
        return [t.as_dict() for t in self.tracks]

    def _associate(self, detections):
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for t, tr in enumerate(self.tracks):
            for d, det in enumerate(detections):
                iou_matrix[t,d] = self.iou(tr.bbox, det)
        # greedy matching
        matches = []
        unmatched_trks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(detections)))
        while True:
            if iou_matrix.size == 0:
                break
            t, d = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            if iou_matrix[t,d] < self.iou_threshold:
                break
            matches.append((t,d))
            iou_matrix[t,:] = -1
            iou_matrix[:,d] = -1
            unmatched_trks.remove(t)
            unmatched_dets.remove(d)
        return matches, unmatched_dets, unmatched_trks
