import cv2
import time
import subprocess

def test_pipeline(pipe_name, pipe_str):
    print(f"👉 Testing: {pipe_name}")
    print(f"   Pipeline: {pipe_str}")
    cap = cv2.VideoCapture(pipe_str, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("   ❌ Failed to open pipeline.")
        return False
    
    success = False
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"   ✅ SUCCESS! Frame received. Shape: {frame.shape}")
            success = True
            break
        time.sleep(0.1)
    
    cap.release()
    if not success:
        print("   ⚠️ Opened but NO FRAMES.")
    return success

print("📸 STARTING CAMERA BRUTE FORCE TEST v2 (1080p)")

# 1. Try NV12 1920x1080 (Safe Memory Size)
test_pipeline("NV12 1080p 30fps", "v4l2src device=/dev/video0 ! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! videoconvert ! appsink")

# 2. Try NV12 640x480 (Safety Low Res)
test_pipeline("NV12 640x480 30fps", "v4l2src device=/dev/video0 ! video/x-raw,format=NV12,width=640,height=480,framerate=30/1 ! videoconvert ! appsink")

# 3. Try Generic Auto-negotiation
test_pipeline("Generic Auto", "v4l2src device=/dev/video0 ! videoconvert ! appsink")

# 4. Try DMABuf (IO-Mode 4) on 1080p
test_pipeline("DMABuf 1080p Auto", "v4l2src device=/dev/video0 io-mode=4 ! video/x-raw,format=NV12,width=1920,height=1080,framerate=30/1 ! videoconvert ! appsink")

print("🏁 TEST COMPLETE")
