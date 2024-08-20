import sys
import subprocess

def install_packages():
    print("Installing required packages...")
    packages = [
        "torch",
        "roboflow",
        "yolov5"
    ]
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully.")

def main():
    # Install packages
    install_packages()

    print("Installation complete. You can now run the data setup script.")

if __name__ == "__main__":
    main()