
# DRONE DEPLOYMENT & SETUP SCRIPT
# Author: Antigravity
# Run this on WINDOWS (PowerShell)

$ErrorActionPreference = "Stop"

# --- CONFIGURATION (CHECK THIS!) ---
$RADXA_USER = "shreyash"
$RADXA_IP = "192.168.0.11"  # <--- CONFIRM THIS IP!
$LOCAL_ROOT = "C:\Users\adish\.gemini\antigravity\scratch\drone_project"

Write-Host ">>> STARTING RADXA DEPLOYMENT..." -ForegroundColor Cyan
Write-Host "Target: $RADXA_USER@$RADXA_IP" -ForegroundColor Gray

# 1. Prepare Staging Area on Radxa
Write-Host ">>> Preparing Staging Area (/tmp/stage)..." -ForegroundColor Yellow
ssh ${RADXA_USER}@${RADXA_IP} "mkdir -p /tmp/stage/raxda_bridge"

# 2. Copy Files to Staging (RAM is always Writable)
Write-Host ">>> Sending Bridge Code..." -ForegroundColor Yellow
scp "$LOCAL_ROOT\raxda_bridge\real_bridge_service.py" ${RADXA_USER}@${RADXA_IP}:/tmp/stage/raxda_bridge/

Write-Host ">>> Sending Setup Script to /tmp/..." -ForegroundColor Yellow
scp "$LOCAL_ROOT\FINAL_RADXA_SETUP.sh" ${RADXA_USER}@${RADXA_IP}:/tmp/FINAL_RADXA_SETUP.sh

# 3. Execute Setup (The Magic)
Write-Host ">>> Executing Final Setup on Radxa..." -ForegroundColor Magenta
Write-Host "PLEASE ENTER PASSWORD IF PROMPTED (Both for SSH and sudo)" -ForegroundColor Red
# We run from /tmp because home might be Read-Only
ssh -t ${RADXA_USER}@${RADXA_IP} "chmod +x /tmp/FINAL_RADXA_SETUP.sh && sudo /tmp/FINAL_RADXA_SETUP.sh"

Write-Host ">>> DEPLOYMENT FINISHED!" -ForegroundColor Green
Write-Host "The Radxa should now be running the bridge code from the SD CARD." -ForegroundColor Green
