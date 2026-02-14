@echo off
echo ========================================================
echo   DEPLOYING FULL BRIDGE TO RADXA (SENSORS + WIFI)
echo ========================================================
echo.
echo Target: 192.168.0.11 (Radxa Zero 3)
echo Files: real_bridge_service.py + run_clean.sh
echo.
echo NOTE: Enter password for 'shreyash' if prompted.
echo.
scp "C:\Users\adish\.gemini\antigravity\scratch\drone_project\raxda_bridge\real_bridge_service.py" shreyash@192.168.0.11:/mnt/sdcard/drone_project/raxda_bridge/
scp "C:\Users\adish\.gemini\antigravity\scratch\drone_project\raxda_bridge\run_clean.sh" shreyash@192.168.0.11:/mnt/sdcard/drone_project/raxda_bridge/
echo.
echo ========================================================
echo   UPLOAD COMPLETE!
echo   Run 'sudo systemctl restart drone-bridge' on Radxa.
echo ========================================================
echo.
pause
