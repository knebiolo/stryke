@echo off
setlocal

REM Launch the Stryke web app locally on Windows
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0Stryke\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

set "FLASK_APP=webapp/app.py"
set "FLASK_SECRET_KEY=dev-secret-key"
set "APP_PASSWORD=expensive5rudabega!@1"

echo Starting Stryke web app...
echo.
echo Open your browser to: http://127.0.0.1:5000
echo Password: %APP_PASSWORD%
echo.

start "" http://127.0.0.1:5000
"%PYTHON_EXE%" -m flask run --host=127.0.0.1 --port=5000

pause
