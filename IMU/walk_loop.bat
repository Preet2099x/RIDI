@echo off
cd /d "%~dp0tools"
python walk.py --closed %*
pause
