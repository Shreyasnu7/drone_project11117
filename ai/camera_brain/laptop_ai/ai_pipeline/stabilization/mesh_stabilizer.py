
import cv2
import numpy as np

class MeshStabilizer:
    """
    Simple Digital Video Stabilization (EIS)
    Uses Good Features to Track + Optical Flow to compute global drift.
    """
    def __init__(self):
        self.prev_gray = None
        self.camera_path = [] # Accumulated transforms
        self.smooth_path = [] # Low-pass filtered path
        self.alpha = 0.1      # Smoothing factor
        
    def process(self, frame: np.ndarray) -> np.ndarray:
        if frame is None: return None
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return frame
            
        # 1. Optical Flow (Example: Good Features)
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        
        if prev_pts is None:
            self.prev_gray = curr_gray
            return frame
            
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, prev_pts, None)
        
        # 2. Filter Valid Points
        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]
        
        if len(prev_pts) < 10:
             self.prev_gray = curr_gray
             return frame
             
        # 3. Estimate Rigid Transform (Translation + Rotation)
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        
        if m is None:
            self.prev_gray = curr_gray
            return frame
            
        # 4. Warp Frame (Inverse Transform to stabilize)
        # Note: A full implementation would accumulate & smooth transforms.
        # This is a simplified "Frame-to-Frame" stabilizer for demo.
        dx = m[0,2]
        dy = m[1,2]
        da = np.arctan2(m[1,0], m[0,0])
        
        # Inverse
        m_inv = cv2.invertAffineTransform(m)
        frame_stab = cv2.warpAffine(frame, m_inv, (frame.shape[1], frame.shape[0]))
        
        self.prev_gray = curr_gray
        return frame_stab
