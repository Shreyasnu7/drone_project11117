# PUSH EVERYTHING TO RADXA (Windows PowerShell)
# Author: Antigravity
# Usage: ./PUSH_AND_RUN.ps1

$ErrorActionPreference = "Stop"

# --- CONFIGURATION (User Provided) ---
$RADXA_USER = "shreyash"
$RADXA_IP = "192.168.0.11"
$LOCAL_ROOT = "C:\Users\adish\.gemini\antigravity\scratch\drone_project"

Write-Host ">>> STARTING FULL RADXA DEPLOYMENT..." -ForegroundColor Cyan

# 1. Create Remote Directories (STAGING in /tmp for RAM access)
Write-Host ">>> Creating Staging folders in /tmp..." -ForegroundColor Yellow
ssh ${RADXA_USER}@${RADXA_IP} "rm -rf /tmp/stage; mkdir -p /tmp/stage/raxda_bridge"

# 2. Transfer Bridge Code (Recursive) to STAGING (/tmp)
Write-Host ">>> Uploading Bridge Code to STAGING..." -ForegroundColor Green
scp -r "$LOCAL_ROOT\raxda_bridge\*" ${RADXA_USER}@${RADXA_IP}:/tmp/stage/raxda_bridge/

# 3. Transfer Setup Script to /tmp
Write-Host ">>> Uploading Setup Script..." -ForegroundColor Green
scp "$LOCAL_ROOT\FINAL_RADXA_SETUP.sh" ${RADXA_USER}@${RADXA_IP}:/tmp/FINAL_RADXA_SETUP.sh

# 4. Make Executable
ssh ${RADXA_USER}@${RADXA_IP} "chmod +x /tmp/FINAL_RADXA_SETUP.sh"

# 5. EXECUTE SETUP (Interactive - May ask for Sudo Password)
Write-Host ">>> RUNNING REMOTE SETUP (You may need to type 'shreyash' password)..." -ForegroundColor Magenta
ssh -t ${RADXA_USER}@${RADXA_IP} "sudo /tmp/FINAL_RADXA_SETUP.sh"

Write-Host ">>> DEPLOYMENT SUCCESSFUL!" -ForegroundColor Cyan
Write-Host ">>> Setup Complete. Service 'drone-bridge' is running."
