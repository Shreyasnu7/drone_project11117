import sys
import os
import traceback

# Add the current directory to sys.path to ensure we can import laptop_ai
sys.path.append(os.getcwd())

def test_import(module_name, class_name=None):
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        print(f"✅ Imported {module_name}")
        if class_name:
            cls = getattr(module, class_name)
            print(f"   Found class {class_name}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        return False
    except AttributeError as e:
        print(f"❌ Failed to find class {class_name} in {module_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error importing {module_name}: {e}")
        traceback.print_exc()
        return False

def verify_director_core():
    print("\n--- Verifying Director Core ---")
    try:
        from laptop_ai.director_core import DirectorCore
        print("✅ Imported DirectorCore class")
        # We won't instantiate it fully as it might need hardware, but we can check if imports inside it invoke successfully
        # or we could try a mock init if possible. 
        # For now, just import tests dependencies.
        return True
    except ImportError as e:
        print(f"❌ Failed to import DirectorCore: {e}")
        traceback.print_exc()
        return False

def main():
    print("Starting AI Module Verification...")
    print(f"CWD: {os.getcwd()}")
    print(f"Python Path: {sys.path}")

    critical_modules = [
        ("laptop_ai.director_core", "DirectorCore"),
        ("laptop_ai.ai_camera_brain", "AICameraBrain"),
        ("laptop_ai.ai_frame_blender", "AIFrameBlender"),
        ("laptop_ai.ai_gimbal_brain", "AIGimbalBrain"),
        ("laptop_ai.execution_router", "ExecutionRouter"),
        ("laptop_ai.ai_shot_planner", "AIShotPlanner"),
        ("laptop_ai.ai_drone_stabilizer", "AIDroneStabilizer"),
        ("laptop_ai.render_master", "RenderMaster"),
        ("laptop_ai.shot_metadata", "ShotMetadataHandler"),
        ("laptop_ai.global_tone_curve", "GlobalToneCurve"),
        ("laptop_ai.ai_color_engine", "AIColorEngine"),
        ("laptop_ai.autonomous_director", "AutonomousDirector"),
        ("laptop_ai.vision_tracker", "VisionTracker"),
        ("laptop_ai.camera_fusion", "FusionEngine") 
    ]

    success_count = 0
    for mod, cls in critical_modules:
        if test_import(mod, cls):
            success_count += 1
    
    print(f"\nVerification Complete: {success_count}/{len(critical_modules)} modules wired successfully.")

if __name__ == "__main__":
    main()
