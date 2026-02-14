# 🚁 HOW TO RUN EVERYTHING: THE FINAL GUIDE

You have a multi-brain system. Here is the exact order to turn it on.

## 1. 🖥️ SERVER (The Heartbeat)
Ensure your Python server is running to relay messages.
```bash
cd drone_server
python main.py
```
*Wait for: `Uvicorn running on http://0.0.0.0:8000`*

## 2. 📡 RADXA BRIDGE (The Nervous System)
This runs **ON THE DRONE**.
1. **Connect**: Power on the drone. Connect to its WiFi or ensure it's on your network.
2. **Deploy**: Run the `DEPLOY_COMMANDS.ps1` script to send the latest `real_bridge_service.py`.
3. **Verify**: SSH into Radxa and check logs.
```bash
ssh rock@<RADXA_IP>
journalctl -u drone-bridge -f
```
*Wait for: `✅ BRIDGE STARTED`, `🔌 MAVLink Connected`, `✅ Cloud Connected!`*

## 3. 🧠 LAPTOP AI (The Brain & Eyes)
This runs **ON YOUR LAPTOP** (connecting to Drone Camera).
1. **Camera**: Ensure valid camera source (Webcam 0 or RTSP stream).
2. **Start**:
```bash
cd ai/camera_brain/laptop_ai
python director_core.py
```
*Wait for: `📷 AI Camera Pipeline: Initializing Full Stack...`, `✅ Color Engine Wired`, `✅ Gemini Strategy Received.`*

## 4. 📱 APP ( The Controller)
1. **Open**: Flutter App on Android/iOS.
2. **Connect**: Enter Server IP.
3. **Check UI**:
   - **Telemetry**: Update rate > 10Hz.
   - **Video**: Low Latency Stream active.
   - **AI Status**: "ONLINE".

## 🛠️ DEBUGGING SAFETY
- **Proximity Alert**: If you put your hand <50cm from sensors, the drone **WILL STOP** and refuse AI commands. This is normal.
- **Cinematic Mode**: If you wave rapidly, the `AIAutoEditor` will trigger "Action Mode" and slow the drone down. This is feature, not bug.
- **Style**: To change look, say "Post Apocalyptic" to the AI.

## 🛑 EMERGENCY
- **LANDING**: Press LAND on App.
- **KILL**: Disarm via Joystick (Down-Left) or Hardware Switch.
- **SYSTEM HALT**: SSH `sudo systemctl stop drone-bridge`.

## ⚠️ READ-ONLY FILESYSTEM (SMART HANDLING)
I have updated `FINAL_RADXA_SETUP.sh` to automatically handle your safety lock:
1.  **Attempts `mount -o remount,rw /`** (The "Open Lock" method you used before).
2.  If that works, it installs and then **remounts RO** ("Close Lock").
3.  If OverlayFS blocks that, it uses `overlayroot-chroot` to persist changes to disk while updating RAM.
**Result**: You run `PUSH_AND_RUN.ps1`, and it just works. Safety remains ON.
