import os
import shutil

print("--- Cleaning __pycache__ ---")
root_dir = os.path.dirname(os.path.abspath(__file__))

count = 0
for root, dirs, files in os.walk(root_dir):
    for d in dirs:
        if d == "__pycache__":
            path = os.path.join(root, d)
            print(f"Removing: {path}")
            shutil.rmtree(path)
            count += 1

print(f"--- Done. Removed {count} cache folders. ---")
