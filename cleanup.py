"""
Cleanup script to remove __pycache__ directories and optimize project size.
"""
import os
import shutil
from pathlib import Path

def remove_pycache():
    """Remove all __pycache__ directories and .pyc files."""
    project_root = Path(__file__).parent
    
    # Remove __pycache__ directories
    for pycache_dir in project_root.rglob("__pycache__"):
        print(f"Removing {pycache_dir}")
        shutil.rmtree(pycache_dir)
    
    # Remove .pyc files
    for pyc_file in project_root.rglob("*.pyc"):
        print(f"Removing {pyc_file}")
        os.remove(pyc_file)

if __name__ == "__main__":
    remove_pycache()
    print("Cleanup complete!") 