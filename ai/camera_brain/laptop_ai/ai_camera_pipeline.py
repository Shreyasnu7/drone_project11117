import cv2
import numpy as np
import os
import sys
import time

# Import ALL Cinematic Modules with Fallbacks
try:
    from laptop_ai.ai_lensfix import LensCorrector 
except: LensCorrector = None

try:
    from laptop_ai.ai_deblur import AIDeblur
except: AIDeblur = None

try:
    from laptop_ai.ai_hdr_engine import AIHdrEngine
except: AIHdrEngine = None

try:
    from laptop_ai.ai_color_engine import AIColorEngine
except: AIColorEngine = None

try:
    from laptop_ai.ai_superres import AISuperRes
except: AISuperRes = None

class AICameraPipeline:
    """
    The Grand Unifier.
    Executes the full cinematic stack:
    Lens -> Deblur -> HDR -> Color -> SuperRes
    """
    def __init__(self):
        print("📷 AI Camera Pipeline: Initializing Full Stack...")
        self.lens = LensCorrector() if LensCorrector else None
        self.deblur = AIDeblur() if AIDeblur else None
        self.hdr = AIHdrEngine() if AIHdrEngine else None
        self.color = AIColorEngine() if AIColorEngine else None
        self.superres = AISuperRes() if AISuperRes else None
        
        if self.color:
             print("✅ Color Engine Wired (Ready for 1000+ LUTs)")

    def load_cinematic_library(self, library_paths):
        """Pass library to Color Engine"""
        if self.color and hasattr(self.color, 'load_library'):
             self.color.load_library(library_paths)

    def process(self, frame):
        """
        Runs the full image processing chain.
        """
        if frame is None: return None
        
        # 1. Lens Correction
        if self.lens:
            try: frame = self.lens.correct(frame)
            except: pass
            
        # 2. Deblur (Motion Compensation)
        if self.deblur:
            try: frame = self.deblur.process(frame)
            except: pass
            
        # 3. HDR (Tone Mapping)
        if self.hdr:
            try: frame = self.hdr.process(frame)
            except: pass
            
        # 4. Color Grading (The Big One)
        if self.color:
            try:
                # Analyze & Grade
                stats = self.color.analyze(frame)
                grade = self.color.propose_grade(stats)
                frame = self.color.apply_grade(frame, grade)
            except: pass
            
        # 5. Super Resolution (Upscale)
        if self.superres:
            try: frame = self.superres.upscale(frame)
            except: pass
            
        return frame
