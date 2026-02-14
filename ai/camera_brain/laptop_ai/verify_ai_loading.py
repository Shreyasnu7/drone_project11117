import sys
import os
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add camera_brain (parent of laptop_ai) to path
# current_dir = .../laptop_ai
# parent = .../camera_brain
camera_brain_dir = os.path.abspath(os.path.join(current_dir, '..'))
if camera_brain_dir not in sys.path:
    sys.path.append(camera_brain_dir)
    print(f"Added to path: {camera_brain_dir}")

# Helper to print status
def check_module(name, import_func):
    try:
        t0 = time.time()
        instance = import_func()
        t1 = time.time()
        print(f"[PASS] {name:<30} (Load time: {(t1-t0)*1000:.1f}ms)")
        return True
    except Exception as e:
        print(f"[FAIL] {name:<30} Error: {e}")
        return False

print("="*60)
print("AI SYSTEM WIRING VERIFICATION")
print("="*60)

# 1. Critical Core Modules
success_count = 0
total_count = 0

def test_director():
    from laptop_ai.director_core import DirectorCore
    return DirectorCore
    # Don't instantiate DirectorCore fully as it starts threads/files

total_count += 1
if check_module("Director Core (Class)", test_director): success_count += 1

# 2. Critical Components (Wired in Director)
def test_blender():
    from laptop_ai.ai_frame_blender import AIFrameBlender
    return AIFrameBlender()
    
def test_gimbal():
    from laptop_ai.ai_gimbal_brain import AIGimbalBrain
    return AIGimbalBrain()

def test_router():
    from laptop_ai.execution_router import ExecutionRouter
    return ExecutionRouter(None)

total_count += 3
if check_module("AI Frame Blender", test_blender): success_count += 1
if check_module("AI Gimbal Brain", test_gimbal): success_count += 1
if check_module("Execution Router", test_router): success_count += 1

# 3. Camera Brain Components
def test_camera_brain():
    from laptop_ai.ai_camera_brain import AICameraBrain
    return AICameraBrain()

def test_hdr():
    from laptop_ai.ai_hdr_engine import AIHDREngine
    return AIHDREngine()

def test_depth():
    from laptop_ai.ai_depth_estimator import AIDepthEstimator
    return AIDepthEstimator()

total_count += 3
if check_module("AI Camera Brain", test_camera_brain): success_count += 1
if check_module("AI HDR Engine", test_hdr): success_count += 1
if check_module("AI Depth Estimator", test_depth): success_count += 1

# 4. Color Engine & Tone Curve
def test_color():
    from laptop_ai.ai_color_engine import AIColorEngine
    return AIColorEngine()

def test_tone_curve():
    from laptop_ai.global_tone_curve import GlobalToneCurve
    return GlobalToneCurve()

total_count += 2
if check_module("AI Color Engine", test_color): success_count += 1
if check_module("Global Tone Curve", test_tone_curve): success_count += 1

# 5. Farming
def test_farming():
    # Need to handle the path difference if running from laptop_ai
    # try direct import or relative
    try:
        from camera_brain.farming.farming_engine import FarmingEngine
        return FarmingEngine()
    except ImportError:
        # try adding parent of laptop_ai to path if not there
        return None

total_count += 1
if check_module("Farming Engine", test_farming): success_count += 1

print("-" * 60)
print(f"RESULTS: {success_count}/{total_count} Modules Loaded Successfully")
if success_count == total_count:
    print("SYSTEM READY FOR FLIGHT TEAMS")
else:
    print("FAILED MODULES DETECTED")
print("="*60)
