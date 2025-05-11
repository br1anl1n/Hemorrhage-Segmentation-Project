import os
import fileinput
import sys

def fix_paths_in_file(file_path):
    """Replace paths in the code files"""
    old_path = "/content/hemorrhage_project/hemorrhage-project"
    new_path = "/content/hemorrhage_project/hemorrhage-project"

    with fileinput.FileInput(file_path, inplace=True) as file:
        for line in file:
            print(line.replace(old_path, new_path), end='')

    print(f"Updated paths in {file_path}")

# Find and update all Python files
for root, dirs, files in os.walk('/content/hemorrhage_project'):
    for file in files:
        if file.endswith('.py'):
            file_path = os.path.join(root, file)
            fix_paths_in_file(file_path)

print("Path updates completed!")
