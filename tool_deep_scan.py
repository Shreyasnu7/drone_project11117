import os
import ast

ROOT_DIR = r"c:\Users\adish\.gemini\antigravity\scratch\drone_project"
SKIP_DIRS = {"venv", "__pycache__", ".git"}

def check_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError in {os.path.basename(filepath)}: {e}"
    except Exception as e:
        return False, f"Error reading {os.path.basename(filepath)}: {e}"

print(f"STARTING DEEP SYNTAX AUDIT in {ROOT_DIR}...")
files_checked = 0
errors = []

for root, dirs, files in os.walk(ROOT_DIR):
    # Skip venv
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            # print(f"Checking {file}...")
            ok, msg = check_file(path)
            files_checked += 1
            if not ok:
                errors.append(msg)
                print(f"FAIL: {msg}")

print("\n" + "="*40)
print(f"COMPLETED. Scanned {files_checked} Python files.")
if errors:
    print(f"FOUND {len(errors)} SYNTAX ERRORS:")
    for e in errors:
        print(e)
else:
    print("ALL FILES PASSED SYNTAX CHECK.")
