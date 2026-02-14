
import cv2
import numpy as np
import math
from laptop_ai.color_components import (
    LOGGammaGenerator, HighlightRecovery, ShadowLiftEngine, 
    LocalColorOptimizer, SaturationEngine, ColorTemperatureEstimator, 
    AdaptiveWhiteBalanceEngine, FilmGrainSimulator, ShotMoodController
)
# Compatibility alias if needed
try:
    from laptop_ai.ai_exposure_engine import ExposureEnginePro
except ImportError:
    ExposureEnginePro = None

# ================================================================
# 3497. Vignette Engine – lens-character artistic shading
# ================================================================

class VignetteEngine:
    """
    Applies soft radial vignette to mimic cinema lenses and guide
    viewer attention. Used lightly unless user explicitly requests.
    """

    def __init__(self, strength=0.25):
        self.strength = strength

    # ------------------------------------------------------------
    # 3511. Apply vignette
    # ------------------------------------------------------------
    def apply(self, frame):
        import numpy as np
        h, w = frame.shape[:2]

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]

        # Center
        cy, cx = h / 2, w / 2

        # Normalized radial distance from center
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        dist /= dist.max()

        # Vignette mask: 1→center, decreasing outward
        mask = 1 - self.strength * (dist**2)
        mask = mask.clip(0.0, 1.0)

        mask = np.repeat(mask[:, :, None], 3, axis=2)

        out = frame.astype(float) * mask
        out = out.clip(0, 255).astype("uint8")

        return out


# ================================================================
# 3549. Final Unified Color Pipeline – puts all modules together
# ================================================================

class AIColorPipeline:
    """
    A complete multi-stage cinematic color grading pipeline that:
        - Converts to LOG
        - Applies highlight/shadow recovery
        - Adjusts white balance + color temperature
        - Applies LUT mood grading
        - Local color correction
        - Saturation tuning
        - Film grain
        - Vignette
    """

    def __init__(self):
        self.log = LOGGammaGenerator()
        self.rec709 = LOGGammaGenerator() # Acts as decoder too with reverse method call
        self.hrec = HighlightRecovery()
        self.srec = ShadowLiftEngine()
        self.lco = LocalColorOptimizer()
        self.sat = SaturationEngine()
        self.temp_est = ColorTemperatureEstimator()
        self.awb = AdaptiveWhiteBalanceEngine()
        self.grain = FilmGrainSimulator()
        self.vignette = VignetteEngine()
        self.mood = ShotMoodController()

    # ------------------------------------------------------------
    # 3584. Run full color pipeline
    # ------------------------------------------------------------
    def process(self, frame, user_text, scene_type="cinematic"):
        # 1. Estimate color temp
        temp = self.temp_est.estimate(frame)
        frame = self.awb.apply(frame, temp)

        # 2. LOG encode
        log_frame = self.log.to_log(frame)

        # 3. Recover highlights + shadows
        log_frame = self.hrec.recover(log_frame)
        log_frame = self.srec.lift(log_frame)

        # 4. Local tile optimizations
        log_frame = self.lco.optimize(log_frame)

        # 5. Color mood (warm/cold/etc.)
        mood = self.mood.decide_mood(user_text, scene_type)
        if mood == "warm":
            log_frame[:, :, 2] = np.clip(log_frame[:, :, 2] * 1.1, 0, 255)
        elif mood == "cold":
            log_frame[:, :, 0] = np.clip(log_frame[:, :, 0] * 1.1, 0, 255)

        # 6. Saturation
        log_frame = self.sat.apply(log_frame)

        # 7. Grain + Vignette
        log_frame = self.grain.apply(log_frame)
        log_frame = self.vignette.apply(log_frame)

        # 8. Convert back to Rec709
        # (Assuming .to_rec709 works on LOG-like input)
        out = self.rec709.to_rec709(log_frame)

        return {
            "frame": out,
            "temperature_k": temp,
            "mood": mood
        }

# ==========================================
# Compatibility Alias for AI Camera Brain
# ==========================================
AIExposureEngine = ExposureEnginePro
PipelineAssembler = AIColorPipeline
