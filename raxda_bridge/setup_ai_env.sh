#!/bin/bash
# P2.3 AI Pipeline Setup Script
# Installs dependencies for Laptop AI / Director Core on Radxa X4

echo "🔧 SETTING UP AI PIPELINE ENVIRONMENT..."

# 1. System Dependencies (APT) - Faster than compiling/pip
echo "📦 Installing System Libraries..."
sudo apt update
sudo apt install -y python3-opencv python3-numpy python3-aiohttp libgl1-mesa-glx

# 2. Python Dependencies (PIP)
echo "🐍 Installing Python Packages..."

# Core AI & Logic
sudo python3 -m pip install openai google-generativeai --break-system-packages

# API & Networking
sudo python3 -m pip install fastapi uvicorn websockets aiofiles --break-system-packages

# Vision (YOLOv8)
echo "👁️ Installing YOLOv8 (Ultralytics)..."
sudo python3 -m pip install ultralytics --break-system-packages

# Pytorch (CPU Version for ARM/Radxa - Critical for disk space/speed)
echo "🔥 Installing PyTorch (CPU Version)..."
sudo python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --break-system-packages

# 3. Validation
echo "✅ Verifying Installation..."
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python3 -c "import torch; print(f'Torch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python3 -c "import ultralytics; print(f'YOLO: {ultralytics.__version__}')"
python3 -c "import google.generativeai; print('Gemini SDK Installed')"

echo "🎉 AI ENVIRONMENT SETUP COMPLETE!"
echo "👉 You can now start the service with: sudo systemctl start ai_director.service"
