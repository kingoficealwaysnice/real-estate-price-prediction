#!/usr/bin/env python3
"""
Real Estate Price Prediction - Application Startup Script
This script handles the initial setup and runs the web application.
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'pandas', 'numpy', 'sklearn', 
        'xgboost', 'tensorflow', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install dependencies using: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_models():
    """Check if trained models exist"""
    model_files = [
        'models/random_forest_model.pkl',
        'models/xgboost_model.pkl',
        'models/linear_regression_model.pkl',
        'models/neural_network_model.h5',
        'models/preprocessing_pipeline.pkl'
    ]
    
    missing_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ {model_file}")
        else:
            missing_models.append(model_file)
            print(f"❌ {model_file} - Missing")
    
    return len(missing_models) == 0

def train_models():
    """Train the machine learning models"""
    print("\n🚀 Training machine learning models...")
    print("This may take a few minutes...")
    
    try:
        result = subprocess.run([sys.executable, 'train_models.py'], 
                              capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ Models trained successfully!")
            return True
        else:
            print("❌ Model training failed!")
            print("Error output:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Model training timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        return False

def start_application():
    """Start the Flask application"""
    print("\n🌐 Starting Real Estate Price Prediction Application...")
    
    # Add src directory to Python path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from src.main import app
        
        # Start the application
        print("✅ Application started successfully!")
        print("🌐 Web interface: http://localhost:5001")
        print("📊 API documentation: http://localhost:5001/api/health")
        print("\nPress Ctrl+C to stop the application")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5001')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Run the Flask app
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except ImportError as e:
        print(f"❌ Error importing application: {e}")
        print("Make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False

def main():
    """Main startup function"""
    print("🏠 Real Estate Price Prediction System")
    print("=" * 50)
    
    # Check Python version
    print("1. Checking Python version...")
    check_python_version()
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    # Create directories
    print("\n3. Creating directories...")
    create_directories()
    
    # Check if models exist
    print("\n4. Checking trained models...")
    models_exist = check_models()
    
    if not models_exist:
        print("\nTrained models not found. Training new models...")
        if not train_models():
            print("\n❌ Failed to train models. Please check the error messages above.")
            print("You can try running 'python train_models.py' manually.")
            sys.exit(1)
    
    # Start application
    print("\n5. Starting application...")
    start_application()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)