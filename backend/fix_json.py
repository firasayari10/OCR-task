"""Fix JSON syntax errors in drugs.json"""
import re
import json

# Read file
with open('C:/Users/Firas/Desktop/ocr/drugs.json', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix unquoted strings in arrays - more comprehensive pattern
patterns = [
    # Pattern 1: ["text", unquoted]
    (r'(\[\"[^\"]+\",\s*)([a-zA-Z][a-zA-Z0-9\s\-\/\(\)]+)(\s*\])', r'\1"\2"\3'),
    # Pattern 2: ["text", "text", unquoted]
    (r'(,\s*)([a-zA-Z][a-zA-Z0-9\s\-\/\(\)]+)(\s*\])', r',\1"\2"\3'),
    # Pattern 3: ["text", unquoted, ...
    (r'(,\s*)([a-zA-Z][a-zA-Z0-9\s\-\/\(\)]+)(\s*,)', r'\1"\2"\3'),
]

fixed_count = 0
for pattern, replacement in patterns:
    for _ in range(20):  # Apply multiple times
        new_content, count = re.subn(pattern, replacement, content)
        if count == 0:
            break
        content = new_content
        fixed_count += count
        print(f"Fixed {count} unquoted strings with pattern")

print(f"Total fixes applied: {fixed_count}")

# Write fixed content
with open('C:/Users/Firas/Desktop/ocr/drugs.json', 'w', encoding='utf-8') as f:
    f.write(content)

print("JSON file fixed!")

# Validate
try:
    with open('C:/Users/Firas/Desktop/ocr/drugs.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"✓ JSON is valid! {len(data)} drugs loaded.")
except json.JSONDecodeError as e:
    print(f"✗ JSON still has errors: {e}")
    print(f"Error at line {e.lineno}, column {e.colno}")

