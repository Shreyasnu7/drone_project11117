import sys
import os

# Ensure we can import from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("[SEARCH] Starting Wiring Verification...")

modules_to_check = [
    "laptop_ai.director_core",
    "laptop_ai.ai_camera_brain",
    "laptop_ai.autonomous_director",
    "laptop_ai.global_tone_curve",
    "laptop_ai.ai_frame_blender",
    "laptop_ai.ai_gimbal_brain",
    "laptop_ai.execution_router",
    "laptop_ai.ai_camera_pipeline",
    "laptop_ai.ai_fusion_pipeline",
    "laptop_ai.follow_brain",
    "laptop_ai.ai_subject_tracker",
    "laptop_ai.camera_fusion"
]

failed = []
passed = []

for mod in modules_to_check:
    print(f"Testing import: {mod}...", end=" ")
    try:
        __import__(mod)
        print("[OK]")
        passed.append(mod)
    except ImportError as e:
        print(f"[FAILED] ({e})")
        failed.append(mod)
    except Exception as e:
        print(f"[ERROR] ({e})")
        failed.append(mod)

print("-" * 40)
print(f"Results: {len(passed)}/{len(modules_to_check)} passed.")

if not failed:
    print("[SUCCESS] ALL SYSTEMS WIRED SUCCESSFULLY.")
else:
    print(f"[STOP] WIRING INCOMPLETE. Failed: {failed}")
    sys.exit(1)
