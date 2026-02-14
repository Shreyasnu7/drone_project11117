
import re
import os

file_path = r"C:\Users\adish\.gemini\antigravity\scratch\drone_project\ai\camera_brain\laptop_ai\ai_frame_blender.py"

print(f"Processing {file_path}...")

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Read {len(lines)} lines.")

new_lines = []
# Match start of line, optional whitespace, digits, optional space, OPTIONAL pipe, space
# Matches: "030 | ", "887  ", "887 "
pattern = re.compile(r"^\s*\d+\s+(\|\s)?")

count = 0
for line in lines:
    if pattern.match(line):
        cleaned = pattern.sub("", line)
        new_lines.append(cleaned)
        count += 1
    else:
        new_lines.append(line)

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Cleaned {count} prefixed lines.")
