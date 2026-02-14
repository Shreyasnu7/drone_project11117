import os
import ast
import sys
from pathlib import Path

# Configuration
ROOT_DIR = Path(__file__).parent.parent  # ai/camera_brain
LAPTOP_AI_DIR = ROOT_DIR / "laptop_ai"
FARMING_DIR = ROOT_DIR / "farming"
CORE_DIR = ROOT_DIR / "core"

# Entry points to start the crawl from
ENTRY_POINTS = [
    LAPTOP_AI_DIR / "director_core.py",
    LAPTOP_AI_DIR / "ai_camera_brain.py",
    LAPTOP_AI_DIR / "autonomous_director.py"
]

def get_all_python_files(directory):
    files = set()
    for p in directory.rglob("*.py"):
        if "venv" in p.parts or ".git" in p.parts:
            continue
        if "verify" not in p.name and "test" not in p.name:
            files.add(p.resolve())
    return files

def get_imports_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports

def resolve_import_to_file(import_name, current_file_path):
    # This is a simplified resolver. 
    # It assumes project structure: laptop_ai.foo -> laptop_ai/foo.py
    
    parts = import_name.split('.')
    
    # Check relative to ROOT_DIR
    candidates = [
        ROOT_DIR / Path(*parts).with_suffix(".py"),
        ROOT_DIR / Path(*parts) / "__init__.py"
    ]
    
    # Check relative to current file's directory (for local imports)
    if current_file_path:
        candidates.append(current_file_path.parent / Path(*parts).with_suffix(".py"))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
            
    return None

def build_dependency_graph(entry_points):
    visited = set()
    queue = list(entry_points)
    
    while queue:
        current_path = queue.pop(0)
        if current_path in visited:
            continue
            
        visited.add(current_path)
        
        # Get imports
        raw_imports = get_imports_from_file(current_path)
        
        for imp in raw_imports:
            # We only care about internal project imports
            if imp.startswith("laptop_ai") or imp.startswith("camera_brain") or imp.startswith("farming") or imp.startswith("core"):
                resolved = resolve_import_to_file(imp, current_path)
                if resolved and resolved not in visited:
                    queue.append(resolved)
    
    return visited

def main():
    print("Starting Wiring Audit...")
    
    # We call these directory scanning and graph building functions
    # (assuming they are defined above as per previous file content)
    # The key change here is just removing the emojis from the print statements below.
    
    all_files_set = get_all_python_files(ROOT_DIR)
    # Convert to list for consistent sorting/printing if needed, but set logic uses set
    
    print(f"Total Python Files Found: {len(all_files_set)}")
    
    wired_files = build_dependency_graph([p.resolve() for p in ENTRY_POINTS])
    print(f"Wired/Reachable Files: {len(wired_files)}")
    
    unwired_files = all_files_set - wired_files
    
    print("\nPotentially Unwired Files:")
    sorted_unwired = sorted(list(unwired_files), key=lambda x: str(x))
    
    if not sorted_unwired:
        print("  (None)")
    
    for f in sorted_unwired:
        # Filter out __init__ noise if it's just empty or small
        try:
             if f.name == "__init__.py" and f.stat().st_size < 50:
                 continue
        except OSError:
             continue
             
        try:
            rel_path = f.relative_to(ROOT_DIR)
            print(f"  - {rel_path}")
        except ValueError:
            print(f"  - {f} (External/Error)")

    if len(all_files_set) > 0:
        coverage = (len(wired_files) / len(all_files_set)) * 100
        print(f"\nCoverage: {len(wired_files)}/{len(all_files_set)} ({coverage:.1f}%)")
    else:
        print("\nCoverage: 0/0 (0.0%)")

if __name__ == "__main__":
    main()
