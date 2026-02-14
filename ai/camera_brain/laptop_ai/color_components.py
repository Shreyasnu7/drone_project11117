
import cv2
import numpy as np
import math
import time

# ================================================================
# Histogram Equalization & Tone Mapping Module
# ================================================================
class HistogramEqualizer:
    def __init__(self, clip_limit=3.0, tile_grid=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def apply_clahe(self, frame):
        if frame is None: return None
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        return enhanced

    def regional_tone_map(self, frame, blocks=3):
        if frame is None: return None
        h, w = frame.shape[:2]
        bh, bw = h // blocks, w // blocks
        output = frame.copy()
        for i in range(blocks):
            for j in range(blocks):
                y0, y1 = i * bh, (i + 1) * bh
                x0, x1 = j * bw, (j + 1) * bw
                region = frame[y0:y1, x0:x1]
                brightness = np.mean(region)
                if brightness < 80: gamma = 1.6
                elif brightness < 120: gamma = 1.2
                elif brightness < 180: gamma = 1.0
                else: gamma = 0.85
                lut = np.array([((k / 255.0) ** gamma) * 255 for k in range(256)]).astype("uint8")
                output[y0:y1, x0:x1] = cv2.LUT(region, lut)
        return output

class ExposureStabilityFilter:
    def __init__(self, alpha=0.25, window_size=12):
        self.alpha = alpha
        self.window_size = window_size
        self.iso_history = []
        self.shutter_history = []

    def update(self, iso, shutter):
        self.iso_history.append(iso)
        self.shutter_history.append(shutter)
        if len(self.iso_history) > self.window_size: self.iso_history.pop(0)
        if len(self.shutter_history) > self.window_size: self.shutter_history.pop(0)

    def get_smoothed(self):
        if not self.iso_history or not self.shutter_history: return None, None
        iso_smooth = np.average(self.iso_history, weights=self._exp_weights())
        shutter_smooth = np.average(self.shutter_history, weights=self._exp_weights())
        return float(iso_smooth), float(shutter_smooth)

    def _exp_weights(self):
        n = len(self.iso_history)
        weights = np.array([(1 - self.alpha) ** (n - i - 1) for i in range(n)])
        return weights / np.sum(weights)

class HighlightRecovery:
    def __init__(self, knee_start=0.75, knee_strength=0.6):
        self.knee_start = knee_start
        self.knee_strength = knee_strength

    def apply(self, frame):
        if frame is None: return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_norm = v.astype(np.float32) / 255.0
        mask = v_norm > self.knee_start
        excess = v_norm - self.knee_start
        v_norm[mask] = self.knee_start + (excess * (1 - self.knee_strength))
        v_new = np.clip(v_norm * 255.0, 0, 255).astype("uint8")
        hsv_new = cv2.merge((h, s, v_new))
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

class ShadowBoost:
    def __init__(self, boost_strength=1.3, noise_floor=0.08):
        self.boost_strength = boost_strength
        self.noise_floor = noise_floor

    def apply(self, frame):
        if frame is None: return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v_norm = v.astype(np.float32) / 255.0
        shadow_mask = v_norm < 0.35
        v_norm[shadow_mask] *= self.boost_strength
        v_norm = np.clip(v_norm, self.noise_floor, 1.0)
        v_new = (v_norm * 255).astype("uint8")
        hsv_new = cv2.merge((h, s, v_new))
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

class ExposureOptimizer:
    def __init__(self):
        self.eq = HistogramEqualizer()
        self.recover = HighlightRecovery()
        self.shadow = ShadowBoost()

    def process(self, frame):
        if frame is None: return None
        step1 = self.eq.apply_clahe(frame)
        step2 = self.eq.regional_tone_map(step1)
        step3 = self.recover.apply(step2)
        final = self.shadow.apply(step3)
        return final

# ================================================================
# AI Color Engine – Film Emulation, Auto-Grading, LUT Pipeline
# ================================================================
class AIColorEngine:
    def __init__(self):
        self.wb_engine = ColorTemperatureStabilizer()
        self.skin_protect = SkinToneProtector()
        self.lut_manager = FilmLUTManager()
        self.scene_classifier = SceneColorClassifier()
        self.curve_generator = CineCurveGenerator()
        self.last_color_temp = None

    def process(self, frame):
        if frame is None: return None
        frame_wb, temp = self.wb_engine.correct(frame)
        self.last_color_temp = temp
        skin_mask = self.skin_protect.get_mask(frame_wb)
        scene_style = self.scene_classifier.classify(frame_wb)
        curve = self.curve_generator.generate(scene_style)
        curved = self.curve_generator.apply_curve(frame_wb, curve)
        graded = self.lut_manager.apply_lut(curved, scene_style)
        final = self.skin_protect.restore_skin_tones(graded, frame_wb, skin_mask)
        return final

class ColorTemperatureStabilizer:
    def __init__(self, stability_alpha=0.15):
        self.stability_alpha = stability_alpha
        self.previous_temp = None

    def estimate_temp(self, frame):
        pixels = frame.reshape(-1, 3).astype(np.float32)
        avg = np.mean(pixels, axis=0)
        kelvin_estimate = (avg[2] / (avg[0] + 1e-5)) * 5000 + 2000
        return float(np.clip(kelvin_estimate, 2500, 9000))

    def correct(self, frame):
        temp = self.estimate_temp(frame)
        if self.previous_temp is None: stable_temp = temp
        else: stable_temp = (self.previous_temp * (1 - self.stability_alpha)) + (temp * self.stability_alpha)
        self.previous_temp = stable_temp
        r_gain, g_gain, b_gain = self._temp_to_gains(stable_temp)
        gains = np.array([b_gain, g_gain, r_gain]).reshape(1, 1, 3)
        balanced = frame.astype(np.float32) * gains
        balanced = np.clip(balanced, 0, 255).astype("uint8")
        return balanced, stable_temp

    def _temp_to_gains(self, kelvin):
        t = kelvin / 100.0
        if t <= 66: r = 1.0
        else: r = 1.292936 * ((t - 60) ** -0.133204)
        if t <= 66: g = 0.390081 * math.log(t) - 0.631841
        else: g = 1.129891 * ((t - 60) ** -0.075514)
        if t >= 66: b = 1.0
        else:
            if t <= 19: b = 0.0
            else: b = 0.543206 * math.log(t - 10) - 1.196254
        m = max(r, g, b, 1e-6)
        return r/m, g/m, b/m

class SkinToneProtector:
    def get_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower_hsv = np.array([0, 40, 60])
        upper_hsv = np.array([25, 255, 255])
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        lower_ycrcb = np.array([0, 135, 85])
        upper_ycrcb = np.array([255, 180, 135])
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        return mask

    def restore_skin_tones(self, graded_frame, original_frame, mask):
        mask3 = cv2.merge([mask, mask, mask]) / 255.0
        return (graded_frame * (1 - mask3) + original_frame * mask3).astype("uint8")

class FilmLUTManager:
    def __init__(self):
        self.luts = {}
        self._load_default_luts()

    def _load_default_luts(self):
        teal_orange = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            teal_orange[i] = [min(255, i * 0.85), min(255, i * 1.05), min(255, 30 + i * 1.15)]
        natural = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            natural[i] = [min(255, i * 1.02), min(255, i * 1.02), min(255, i * 1.02)]
        night = np.zeros((256, 3), dtype=np.float32)
        for i in range(256):
            night[i] = [min(255, i * 1.25), min(255, i * 1.15), min(255, i)]
        self.luts = {"cinematic_teal_orange": teal_orange, "cinematic_natural": natural, "night_boost": night}

    def apply_lut(self, frame, style):
        lut = self.luts.get("cinematic_teal_orange" if style == "cinematic" else "night_boost" if style == "night" else "cinematic_natural")
        return np.clip(lut[frame], 0, 255).astype("uint8")

class SceneColorClassifier:
    def classify(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        high_ratio = np.sum(hist[230:]) / np.sum(hist)
        low_ratio  = np.sum(hist[:25])  / np.sum(hist)
        if brightness < 55: return "night"
        if high_ratio > 0.15 and low_ratio > 0.15: return "high_contrast"
        return "cinematic"

class CineCurveGenerator:
    def generate(self, scene_type):
        x = np.linspace(0, 1, 256)
        if scene_type == "cinematic": y = 0.5 + 0.25 * np.tanh(2 * (x - 0.5))
        elif scene_type == "high_contrast": y = 0.45 + 0.35 * np.tanh(3 * (x - 0.5))
        elif scene_type == "night": y = x ** 0.8
        else: y = x
        return (y * 255).astype("uint8")

    def apply_curve(self, frame, curve):
        return cv2.LUT(frame, curve)

class LOGGammaGenerator:
    def __init__(self):
        self.a = 0.25; self.b = 0.45; self.c = 0.10
    def to_log(self, frame):
        f = frame.astype(np.float32) / 255.0
        log_frame = (np.log10(1 + self.b * f) / np.log10(1 + self.b)) * (1 - self.c) + self.c
        return np.clip(log_frame * 255, 0, 255).astype("uint8")
    def to_rec709(self, log_frame):
        f = log_frame.astype(np.float32) / 255.0
        rec = ((10 ** (((f - self.c) / (1 - self.c)) * np.log10(1 + self.b))) - 1) / self.b
        return np.clip(rec * 255, 0, 255).astype("uint8")

class ShadowLiftEngine:
    def __init__(self, strength=0.45): self.strength = strength
    def lift(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        v = hsv[:, :, 2] / 255.0
        mask = v < 0.25
        lifted = v.copy()
        lifted[mask] = lifted[mask] ** (1 - self.strength)
        hsv[:, :, 2] = np.clip(lifted * 255, 0, 255)
        return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

class LocalColorOptimizer:
    def __init__(self, tile=32): self.tile = tile
    def optimize(self, frame):
        h, w, _ = frame.shape
        out = frame.copy()
        for y in range(0, h, self.tile):
            for x in range(0, w, self.tile):
                tile = frame[y:y+self.tile, x:x+self.tile]
                if tile.size == 0: continue
                b_avg = tile[:, :, 0].mean(); g_avg = tile[:, :, 1].mean(); r_avg = tile[:, :, 2].mean()
                wb = np.array([g_avg / (b_avg + 1e-5), 1.0, g_avg / (r_avg + 1e-5)])
                corrected = tile.astype(np.float32) * wb
                out[y:y+self.tile, x:x+self.tile] = np.clip(corrected, 0, 255).astype("uint8")
        return out

class SaturationEngine:
    def __init__(self, strength=1.15): self.strength = strength
    def apply(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= self.strength
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)

class ShotMoodController:
    def __init__(self): self.valid_moods = ["cinematic", "warm", "cold", "neutral", "night"]
    def decide_mood(self, user_text, scene_type):
        user_text = user_text.lower()
        if "warm" in user_text: return "warm"
        if "cold" in user_text: return "cold"
        if "night" in user_text: return "night"
        if scene_type in self.valid_moods: return scene_type
        return "cinematic"

class ColorTemperatureEstimator:
    def __init__(self): self.min_temp = 2500; self.max_temp = 9000
    def estimate(self, frame):
        b, g, r = [c.mean() for c in cv2.split(frame.astype(float))]
        g = max(g, 1e-6)
        rg = r / g; bg = b / g
        temp = 6500 + (rg - bg) * 2200
        return float(max(self.min_temp, min(self.max_temp, temp)))

class AdaptiveWhiteBalanceEngine:
    def __init__(self): self.ref_temp = 6500
    def temp_to_gain(self, temp_k):
        delta = (temp_k - self.ref_temp) / 3500.0
        return np.array([1.0 + (delta * 0.6), 1.0, 1.0 + (-delta * 0.5)], dtype=float)
    def apply(self, frame, temp_k):
        gains = self.temp_to_gain(temp_k)
        corrected = frame.astype(float)
        corrected[:, :, 0] *= gains[0]; corrected[:, :, 1] *= gains[1]; corrected[:, :, 2] *= gains[2]
        return corrected.clip(0, 255).astype("uint8")

class FilmGrainSimulator:
    def __init__(self, strength=0.07): self.strength = strength
    def apply(self, frame):
        h, w, _ = frame.shape
        grain = np.random.normal(0, 25, (h, w)).astype(np.float32) * self.strength
        grain_rgb = np.repeat(grain[:, :, None], 3, axis=2)
        return np.clip(frame.astype(np.float32) + grain_rgb, 0, 255).astype("uint8")

class VignetteEngine:
    def __init__(self): pass

class ColorComponents:
    def __init__(self): 
        pass
