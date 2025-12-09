"""
Install SAM2 from source with Python 3.12 compatibility fixes
"""
import subprocess
import sys
import os

def install_sam2():
    """Install SAM2 from source"""
    print("Installing SAM2 from source...")
    print("This may take a few minutes...")
    
    # Upgrade pip and setuptools first
    print("\n1. Upgrading pip and setuptools...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    
    # Clone or update the repository
    sam2_dir = "segment-anything-2"
    if os.path.exists(sam2_dir):
        print(f"\n2. Repository {sam2_dir} already exists. Updating...")
        os.chdir(sam2_dir)
        subprocess.check_call(["git", "pull"])
    else:
        print("\n2. Cloning SAM2 repository...")
        subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/segment-anything-2.git"])
        os.chdir(sam2_dir)
    
    # Install in editable mode
    print("\n3. Installing SAM2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    
    print("\n✓ SAM2 installed successfully!")
    print("\nNext steps:")
    print("1. Download model weights: python download_model.py")
    print("2. Run the server: python main.py")

if __name__ == "__main__":
    try:
        install_sam2()
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Installation failed: {e}")
        print("\nAlternative: Try using Python 3.11 or 3.10 instead of 3.12")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)



