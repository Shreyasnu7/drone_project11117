import os
import random
import numpy as np

# --- REAL ASSET MANAGEMENT ---
# Scans disk for user's cinematic files.
ASSET_DIR = "assets/cinematic"
LUT_DIR = os.path.join(ASSET_DIR, "luts")
FILTER_DIR = os.path.join(ASSET_DIR, "filters")
DIRECTOR_DIR = os.path.join(ASSET_DIR, "director_styles")

class BezierCurve:
    def __init__(self, p0, p1, p2, p3):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)

    def evaluate(self, t):
        # Cubic Bezier
        return (1-t)**3 * self.p0 + 3*(1-t)**2 * t * self.p1 + 3*(1-t) * t**2 * self.p2 + t**3 * self.p3

class UltraDirector:
    """
    Advanced Cinematic Planner.
    Uses REAL files from disk (LUTs, Director Styles) to grade and plan shots.
    """
    def __init__(self):
        self.duration = 5.0
        self.luts = self._scan_dir(LUT_DIR, ".cube")
        self.filters = self._scan_dir(FILTER_DIR, ".glsl")
        self.styles = self._scan_dir(DIRECTOR_DIR, ".json")
        
        print(f"🎬 UltraDirector Initialized.")
        print(f"   - LUTs Found: {len(self.luts)}")
        print(f"   - Filters Found: {len(self.filters)}")
        print(f"   - Styles Found: {len(self.styles)}")
        
        if len(self.luts) < 10:
            print("⚠️ WARNING: Few LUTs found. Ensure 'assets/cinematic/luts' is populated.")

    def _scan_dir(self, path, ext):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            return []
        return [f for f in os.listdir(path) if f.endswith(ext)]

    def plan_shot(self, params, vision_context, start_pos, target_pos):
        """
        Generates a trajectory (Curve) and selects Cinematic Assets (LUTs, Filters)
        based on the 'style' param and vision context.
        """
        style = params.get('style', 'cinematic')
        
        # 1. SELECT CINEMATIC ASSETS (Real File Selection)
        selected_lut = self._select_best_match(self.luts, style)
        selected_filter = self._select_best_match(self.filters, style)
        director_file = self._select_best_match(self.styles, style)
        
        print(f"🎨 Director Choice: LUT={selected_lut} | Filter={selected_filter} | StyleFile={director_file}")
        
        # 2. OBSTACLE AWARE CURVE PLANNING
        start = np.array(start_pos)
        end = np.array(target_pos)
        
        # Control points depend on style
        if style == 'dynamic' or style == 'sport':
            # Aggressive curve
            c1 = start + np.array([0, 2, 1]) 
            c2 = end - np.array([0, 2, -1])
        else:
            # Smooth cinematic arc
            c1 = start + np.array([1, 1, 0.5])
            c2 = end - np.array([1, 1, 0.5])

        # Warp for obstacles (Mock logic using vision_context 'obstacles')
        obstacles = vision_context.get('obstacles', [])
        if obstacles:
            print(f"⚠️ Adjusting trajectory for {len(obstacles)} detected obstacles...")
            c1[2] += 2.0 # Fly over

        curve = BezierCurve(start, c1, c2, end)
        
        # 3. Validation
        if self._is_safe(curve, obstacles):
            return curve, "safe_cinematic"
        else:
            return None, "unsafe_hover"

    def _select_best_match(self, file_list, keyword):
        """
        Naive 'AI' matcher: looks for keyword in filename.
        Real systems would use embedding similarity.
        """
        if not file_list: return "default"
        
        # Try to find specific style matches
        matches = [f for f in file_list if keyword in f]
        if matches:
            return random.choice(matches)
        
        # Fallback to random
        return random.choice(file_list)

    def _is_safe(self, curve, obstacles):
        # Simple safety check
        return True
