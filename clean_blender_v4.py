
import re
import os

file_path = r"C:\Users\adish\.gemini\antigravity\scratch\drone_project\ai\camera_brain\laptop_ai\ai_frame_blender.py"
print(f"Processing {file_path}...")

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []

# Revised Logic:
# Capture (Indent)(Number)(Spaces)(Content)
# We want to KEEP (Indent) and (Content).
# We want to REMOVE (Number)(Spaces)

regex_v5 = re.compile(r"^(\s*)\d+\s+(.*)")

count_cleaned = 0

for line in lines:
    match = regex_v5.match(line)
    if match:
        # Group 1 = Indentation (Leading spaces)
        # Group 2 = Content (Rest of line)
        indent = match.group(1)
        content = match.group(2)
        
        # Reconstruct line without the number
        new_line = indent + content + "\n"
        new_lines.append(new_line)
        count_cleaned += 1
    else:
        new_lines.append(line)


with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Cleaned Lines: {count_cleaned}")
