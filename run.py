"""
Launcher script for the PDF Q&A System.
"""
import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get the absolute path to the project root
    project_root = Path(__file__).resolve().parent
    
    # Add the project root to PYTHONPATH
    python_path = os.environ.get('PYTHONPATH', '')
    if python_path:
        os.environ['PYTHONPATH'] = f"{project_root}{os.pathsep}{python_path}"
    else:
        os.environ['PYTHONPATH'] = str(project_root)
    
    # Set the current working directory to the project root
    os.chdir(project_root)
    
    # Run the Streamlit app using subprocess to inherit the environment
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "app/main.py",
        "--server.address", "localhost"
    ], env=os.environ)

# Run the Streamlit app
if __name__ == "__main__":
    main() 