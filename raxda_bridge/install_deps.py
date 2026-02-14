import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    try:
        subprocess.check_call(cmd, shell=True)
        print("✅ Success")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")

def main():
    print("🔧 FORCE INSTALLING DEPENDENCIES (Python Wrapper)")
    
    # 1. Install System Deps
    run_cmd("sudo apt update")
    run_cmd("sudo apt install -y python3-pip python3-venv")

    # 2. Fix Serial & Install YDLidar
    print("📦 Installing Python Libraries...")
    run_cmd("sudo python3 -m pip install pyserial --break-system-packages --force-reinstall")
    
    print("📦 Building YDLidar SDK from source (PyPI missing for ARM)...")
    run_cmd("sudo apt install -y cmake swig git python3-dev build-essential python3-setuptools")
    
    # Clean previous
    if os.path.exists("YDLidar-SDK"):
        run_cmd("sudo rm -rf YDLidar-SDK")
        
    # Clone & Build
    run_cmd("git clone https://github.com/YDLIDAR/YDLidar-SDK.git")
    os.chdir("YDLidar-SDK")
    run_cmd("mkdir build")
    os.chdir("build")
    run_cmd("cmake ..")
    run_cmd("make")
    run_cmd("sudo make install")
    
    # Python Bindings
    os.chdir("..") # Back to YDLidar-SDK root
    run_cmd("sudo python3 setup.py install")
    os.chdir("..") # Back to raxda_bridge
    
    # 3. Permissions
    print("🔓 Setting Permissions...")
    run_cmd("sudo usermod -a -G dialout shreyash")
    run_cmd("sudo chmod 666 /dev/ttyS*")
    
    print("\n✅ SETUP COMPLETE. Reboot recommended.")

if __name__ == "__main__":
    main()
