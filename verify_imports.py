import os
import sys
import importlib
import traceback

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir))) # Adjust based on location
# Assuming we place this in scratch/drone_project/ai/camera_brain/laptop_ai or similar for testing
# Actually, let's look at where we are. 
# We are likely in C:\Users\adish\.gemini\antigravity\scratch\drone_project
# Let's verify paths dynamically.

def find_project_root(start_path):
    path = start_path
    while path != os.path.dirname(path):
        if "laptop_ai" in os.listdir(path):
             return path
        path = os.path.dirname(path)
    return None

# We'll assume this script is run from the root of the ai logic or we pass it.
# Let's try to add the standard paths.
sys.path.append(os.getcwd())
# Also add 'C:\Users\adish\.gemini\antigravity\scratch\drone_project\ai\camera_brain'
sys.path.append(os.path.join(os.getcwd(), 'ai', 'camera_brain'))

def get_python_files(directory):
    py_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                full_path = os.path.join(root, file)
                module_path = os.path.relpath(full_path, os.getcwd())
                module_name = module_path.replace(os.sep, ".")[:-3]
                py_files.append(module_name)
    return py_files

def verify_modules(modules):
    success = []
    failed = []
    
    print(f"[INFO] Verifying {len(modules)} modules...")
    
    for module in modules:
        try:
            importlib.import_module(module)
            success.append(module)
            # print(f"[OK] {module}")
        except Exception as e:
            failed.append((module, str(e)))
            print(f"[FAIL] {module}: {e}")
            
    return success, failed

if __name__ == "__main__":
    # Target specific directories
    # ADJUSTED: We want to find where 'laptop_ai' etc live.
    # If we are in 'drone_project', then 'ai/camera_brain' contains 'laptop_ai'
    
    base_path = os.getcwd()
    ai_camera_brain_path = os.path.join(base_path, 'ai', 'camera_brain')
    
    print(f"Base path: {base_path}")
    print(f"AI Camera Brain path: {ai_camera_brain_path}")

    if os.path.exists(ai_camera_brain_path):
        sys.path.append(ai_camera_brain_path)
        print(f"Added {ai_camera_brain_path} to sys.path")
    else:
        print(f"Warning: {ai_camera_brain_path} does not exist!")

    # Target finding logic
    target_dirs = [
        os.path.join(ai_camera_brain_path, "laptop_ai"),
        os.path.join(ai_camera_brain_path, "core"),
        os.path.join(ai_camera_brain_path, "farming")
    ]
    
    all_modules = []
    for d in target_dirs:
        if os.path.exists(d):
            # We want relative to ai_camera_brain_path because that is in sys.path
            # e.g. laptop_ai/xyz.py -> laptop_ai.xyz
            for root, _, files in os.walk(d):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, ai_camera_brain_path)
                        module_name = rel_path.replace(os.sep, ".")[:-3]
                        all_modules.append(module_name)
        else:
            print(f"Directory not found: {d}")

    success, failed = verify_modules(all_modules)
    
    print("\n" + "="*50)
    print(f"Summary: {len(success)} passed, {len(failed)} failed")
    print("="*50)
    
    if failed:
        print("\nFailed Modules:")
        for m, err in failed:
            print(f"- {m}: {err}")
