import cv2
import time
import numpy as np
import os
from collections import deque

class AIAutoEditor:
    """
    Real-Time Auto Editor.
    Analyzes "Excitement Level" of the video feed.
    If Excitement > Threshold, saves the clip (Buffered).
    """
    def __init__(self, buffer_seconds=10.0):
        print("🎬 AI Auto-Editor Initialized (Looking for Action...)")
        self.buffer_size = 30 * int(buffer_seconds) # 30fps
        self.buffer = deque(maxlen=self.buffer_size)
        self.is_clipping = False
        self.clip_frames = []
        self.cooldown = 0
        
    def process_frame(self, frame, detections, motion_score):
        """
        Decides if this frame is 'Exciting'.
        """
        # Always buffer
        self.buffer.append(frame)
        
        # Calculate Excitement Score
        score = 0
        # 1. Detections (People/Cars = Interesting)
        if len(detections) > 0:
            score += 5.0 * len(detections)
            
        # 2. Motion (Fast movement = Action)
        score += motion_score * 2.0
        
        # Trigger Limit
        if score > 20.0 and self.cooldown == 0 and not self.is_clipping:
            print(f"🎬 SMART EDIT: Action Detected! (Score={score:.1f}) Starting Clip...")
            self.start_clip()
            
        if self.is_clipping:
            self.clip_frames.append(frame)
            # Stop if boring or too long (10s max after trigger)
            if len(self.clip_frames) > 300: 
                self.save_clip()

        if self.cooldown > 0:
            self.cooldown -= 1
            
    def start_clip(self):
        self.is_clipping = True
        # Include pre-roll buffer
        self.clip_frames = list(self.buffer)
        
    def save_clip(self):
        filename = f"DCIM/SmartEdit_Action_{int(time.time())}.mp4"
        os.makedirs("DCIM", exist_ok=True)
        
        if not self.clip_frames: return
        
        h, w = self.clip_frames[0].shape[:2]
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        for f in self.clip_frames:
            out.write(f)
        out.release()
        
        print(f"💾 SMART EDIT SAVED: {filename}")
        self.is_clipping = False
        self.clip_frames = []
        self.cooldown = 150 # 5s cooldown
