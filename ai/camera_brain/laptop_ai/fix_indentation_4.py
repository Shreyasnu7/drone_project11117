
import os

target_file = r"C:\Users\adish\.gemini\antigravity\scratch\drone_project\ai\camera_brain\laptop_ai\global_tone_curve.py"

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Fix lines around 56546 (indices ~56545)
# Set to 7 spaces.
affected_indices = [56542, 56544, 56545]
target_indent = '       ' # 7 spaces

for idx in affected_indices:
    content = lines[idx].lstrip()
    if content.startswith('#'):
       # Comments at 7 spaces? Or 6? Let's check original.
       # Original was 6 spaces for comments. 
       # If code is at 7 spaces, comments at 6 spaces is technically OK but ugly. 
       # But python doesn't care about comments indentation usually.
       # However, mixing indentation levels often confuses people.
       # I'll set comments to 7 spaces too.
       pass
    lines[idx] = target_indent + content

# Also inspect line 56560 (index 56559)
print(f"Original line 56560: {repr(lines[56559])}")
# If it has 8 spaces, fix to 7.
if lines[56559].startswith('        ') and not lines[56559].startswith('         '):
    # starts with 8 spaces.
    lines[56559] = target_indent + lines[56559].lstrip()
    print("Fixed line 56560 to 7 spaces.")

with open(target_file, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Successfully updated global_tone_curve.py lines around 56546")
