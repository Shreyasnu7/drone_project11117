
import os
import re

FILE_PATH = r"C:\Users\adish\.gemini\antigravity\scratch\drone_project\ai\camera_brain\laptop_ai\ai_frame_blender.py"
BACKUP_PATH = FILE_PATH + ".indent_bak_v4"

def restore_indentation():
    if not os.path.exists(FILE_PATH):
        print("File not found.")
        return

    # Create backup
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
    
    with open(BACKUP_PATH, 'w', encoding='utf-8') as f:
        f.writelines(original_lines)
    
    print(f"Backup created at {BACKUP_PATH}")

    fixed_lines = []
    
    # State
    current_indent_level = 0 
    in_docstring = False
    docstring_char = None # ' or "
    dedented_by_block_end = False # New flag to prevent double dedent for elif/else
    
    for i, raw_line in enumerate(original_lines):
        line = raw_line.strip()
        
        # Skip pure empty lines but preserve correct newlines
        if not line:
            fixed_lines.append("\n")
            # Do NOT reset flag here? 
            # If we return, then empty line, then elif... the empty line shouldn't break the logic.
            # But technically empty lines don't change indent. 
            continue
            
        # --- DOCSTRING DETECTION ---
        triplets_double = line.count('"""')
        triplets_single = line.count("'''")
        
        if not in_docstring:
            if triplets_double > 0 and triplets_double % 2 == 1:
                in_docstring = True
                docstring_char = '"'
            elif triplets_single > 0 and triplets_single % 2 == 1:
                in_docstring = True
                docstring_char = "'"
        else:
            if docstring_char == '"' and triplets_double > 0 and triplets_double % 2 == 1:
                in_docstring = False
                docstring_char = None
            elif docstring_char == "'" and triplets_single > 0 and triplets_single % 2 == 1:
                in_docstring = False
                docstring_char = None

        # --- PRE-PROCESSING (Dedent) ---
        if not in_docstring and docstring_char is None:
            # Classes/Defs/Imports always force level
            if line.startswith("class "):
                current_indent_level = 0
            elif line.startswith("def "):
                current_indent_level = 1
            elif line.startswith("import ") or line.startswith("from "):
                current_indent_level = 0
            
            # ELIF/ELSE/EXCEPT/FINALLY
            # These must satisfy: Level matches the opening block (i.e., dedent from body)
            elif line.startswith("else:") or line.startswith("elif ") or line.startswith("except") or line.startswith("finally:"):
                # If we just dedented because of return/break, DO NOT DEDENT AGAIN.
                if current_indent_level > 1:
                    if not dedented_by_block_end:
                         current_indent_level -= 1
                    else:
                         # We are already dedented, so this line is correct at current level
                         pass

        # --- WRITE LINE ---
        indent_str = "    " * current_indent_level
        fixed_lines.append(indent_str + line + "\n")
        
        # --- POST-PROCESSING (Indent for NEXT line) ---
        dedented_by_block_end = False # Reset flag for next line
        
        if not in_docstring and docstring_char is None:
            code_part = line.split('#')[0].strip()
            
            if code_part.endswith(":"):
                current_indent_level += 1
            
            if current_indent_level > 2:
                if code_part.startswith("return") or code_part.startswith("break") or code_part.startswith("continue") or code_part.startswith("raise ") or code_part == "pass":
                    current_indent_level -= 1
                    dedented_by_block_end = True # Flag that we reduced index

    # Write back
    with open(FILE_PATH, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"Restored indentation for {len(fixed_lines)} lines.")

if __name__ == "__main__":
    restore_indentation()
