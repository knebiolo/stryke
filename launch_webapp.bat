@echo off
setlocal

REM Launch the Stryke webapp from the repo root
cd /d "%~dp0"
python webapp\app.py

endlocal
