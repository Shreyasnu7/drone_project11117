# DRONE SYSTEM DEPLOYMENT SCRIPT
# Author: Antigravity (Google Deepmind)
# Purpose: Transfer updated code to Radxa and verify Laptop AI files.

$ErrorActionPreference = "Stop"

# --- CONFIGURATION ---
$RADXA_USER = "rock"
$RADXA_IP = "192.168.0.11"  # CHANGE THIS to your Radxa's actual IP
$ThePassword = "rock"        # Default, change/secure if needed
$LOCAL_ROOT = "C:\Users\adish\.gemini\antigravity\scratch\drone_project"

Write-Host "🚀 STARTING DRONE DEPLOYMENT..." -ForegroundColor Cyan

# 1. RADXA BRIDGE (The Critical Link)
Write-Host "📡 Deploying Bridge to Radxa ($RADXA_IP)..." -ForegroundColor Yellow
# Using SCP (Ensure OpenSSH Client is installed in Windows)
# NOTE: If prompted for password, it is likely 'rock'
scp "$LOCAL_ROOT\raxda_bridge\real_bridge_service.py" ${RADXA_USER}@${RADXA_IP}:~/drone-bridge/real_bridge_service.py
scp "$LOCAL_ROOT\requirements.txt" ${RADXA_USER}@${RADXA_IP}:~/drone-bridge/requirements.txt

# 2. REMOTE COMMANDS (Update Dependencies & Restart)
Write-Host "🔧 Installing Dependencies & Restarting Service..." -ForegroundColor Yellow
ssh ${RADXA_USER}@${RADXA_IP} "pip3 install -r ~/drone-bridge/requirements.txt && sudo systemctl restart drone-bridge"

# 3. LAPTOP AI (Local Verification)
Write-Host "💻 Verifying Laptop AI Files..." -ForegroundColor Green
$AI_PATH = "$LOCAL_ROOT\ai\camera_brain\laptop_ai"
if (Test-Path "$AI_PATH\director_core.py") { Write-Host "  [OK] director_core.py" } else { Write-Host "  [MISSING] director_core.py" -ForegroundColor Red }
if (Test-Path "$AI_PATH\ai_camera_pipeline.py") { Write-Host "  [OK] ai_camera_pipeline.py" } else { Write-Host "  [MISSING] ai_camera_pipeline.py" -ForegroundColor Red }
if (Test-Path "$AI_PATH\ai_color_engine.py") { Write-Host "  [OK] ai_color_engine.py" } else { Write-Host "  [MISSING] ai_color_engine.py" -ForegroundColor Red }
if (Test-Path "$LOCAL_ROOT\cloud_ai\gemini_director.py") { Write-Host "  [OK] gemini_director.py (API KEY CHECK)" } else { Write-Host "  [MISSING] gemini_director.py" -ForegroundColor Red }

Write-Host "✅ DEPLOYMENT COMPLETE!" -ForegroundColor Cyan
Write-Host "👉 NOW READ: HOW_TO_RUN_EVERYTHING.md" -ForegroundColor White
