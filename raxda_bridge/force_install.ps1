$ErrorActionPreference = "Stop"

Write-Host "🚀 STARTING FORCE DEPLOYMENT..." -ForegroundColor Cyan

# 1. UPLOAD FILE
Write-Host "1. Uploading real_bridge_service.py..."
scp "c:\Users\adish\.gemini\antigravity\scratch\drone_project\raxda_bridge\real_bridge_service.py" shreyash@192.168.0.11:/home/shreyash/
if ($LASTEXITCODE -ne 0) { Write-Error "SCP Failed"; exit }

# 2. REMOTE INSTALL SEQUENCE
Write-Host "2. Installing on Radxa (Interactive - Enter Password if asked)..."
# Use -t to force TTY for sudo password prompt.
# Use single line string to avoid PowerShell CRLF issues passing \r to Linux.
ssh -t shreyash@192.168.0.11 "sudo killall python3; sudo cp /home/shreyash/real_bridge_service.py /mnt/sdcard/drone_project/raxda_bridge/real_bridge_service.py; sudo chmod 777 /mnt/sdcard/drone_project/raxda_bridge/real_bridge_service.py; echo '🔍 VERIFYING...'; grep 'FINAL MAP' /mnt/sdcard/drone_project/raxda_bridge/real_bridge_service.py && echo '✅ VERIFIED!' || echo '❌ UPDATE FAILED'; sudo /mnt/sdcard/drone_project/raxda_bridge/run_clean.sh"

