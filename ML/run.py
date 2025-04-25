#!/usr/bin/env python
"""
AquaSense - AI-based Spatial Analysis for Drinking Water Information
Run script to train models and start the web application
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from time import sleep

def check_data_files():
    """Check if all required data files exist"""
    required_files = [
        "data/water_quality_data.csv",
        "data/Geology Maharashtra.xlsx",
        "data/Lithology Maharashtra.xlsx"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    return missing_files

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "data",
        "models",
        "web/static/plots",
        "web/static/images"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def train_models():
    """Train all machine learning models"""
    print("Training machine learning models...")
    
    # Change to src directory and run model training
    current_dir = os.getcwd()
    os.chdir("src")
    
    try:
        subprocess.run([sys.executable, "model_training.py"], check=True)
        print("Model training completed successfully!")
    except subprocess.CalledProcessError:
        print("Error: Model training failed. Please check the logs.")
        return False
    finally:
        os.chdir(current_dir)
    
    return True

def run_web_app():
    """Run the Flask web application"""
    print("Starting web application...")
    
    # Change to web directory and run Flask app
    current_dir = os.getcwd()
    os.chdir("web")
    
    # Open web browser after a short delay
    def open_browser():
        sleep(2)
        webbrowser.open("http://localhost:5000")
    
    try:
        # Start browser in a separate thread
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the Flask app
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nWeb application stopped.")
    except subprocess.CalledProcessError:
        print("Error: Failed to start web application. Please check the logs.")
    finally:
        os.chdir(current_dir)

def main():
    parser = argparse.ArgumentParser(description="AquaSense - Run script")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--train-only", action="store_true", help="Only train models, don't start web app")
    args = parser.parse_args()
    
    print("=" * 80)
    print("AquaSense - AI-based Spatial Analysis for Drinking Water Information")
    print("=" * 80)
    
    # Create directories
    create_directories()
    
    # Check data files
    missing_files = check_data_files()
    if missing_files:
        print("Error: The following required data files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease add these files to continue.")
        return
    
    # Train models if needed
    if not args.skip_training:
        if not train_models():
            return
    
    # Run web app if not train-only mode
    if not args.train_only:
        run_web_app()

if __name__ == "__main__":
    main() 