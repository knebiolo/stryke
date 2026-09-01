@echo off
setlocal EnableExtensions DisableDelayedExpansion

cd /d "%~dp0"

set "PYTHON_EXE="

rem --- Optional override for admins/power users ---
if defined STRYKE_PYTHON_EXE call :try_python "%STRYKE_PYTHON_EXE%"

rem --- Repo-local virtual env / embedded env ---
call :try_python "%~dp0.venv\Scripts\python.exe"
call :try_python "%~dp0venv\Scripts\python.exe"
call :try_python "%~dp0Stryke\Scripts\python.exe"

rem --- Common conda env names / locations for shared multi-user installs ---
call :try_python "%USERPROFILE%\Desktop\conda_envs\stryke\python.exe"
call :try_python "%USERPROFILE%\miniconda3\envs\stryke\python.exe"
call :try_python "%USERPROFILE%\anaconda3\envs\stryke\python.exe"
call :try_python "%LOCALAPPDATA%\anaconda3\envs\stryke\python.exe"
call :try_python "%LOCALAPPDATA%\miniconda3\envs\stryke\python.exe"
call :try_python "%USERPROFILE%\.conda\envs\stryke\python.exe"
call :try_python "%ProgramData%\anaconda3\envs\stryke\python.exe"
call :try_python "%ProgramData%\miniconda3\envs\stryke\python.exe"

rem --- Already-activated conda/venv session (CONDA_PREFIX / VIRTUAL_ENV) ---
if defined CONDA_PREFIX call :try_python "%CONDA_PREFIX%\python.exe"
if defined VIRTUAL_ENV call :try_python "%VIRTUAL_ENV%\Scripts\python.exe"

rem --- Conda base env, discovered dynamically (covers custom install paths / non-"stryke" env names) ---
if not defined PYTHON_EXE (
  where conda >nul 2>nul
  if not errorlevel 1 (
    for /f "delims=" %%I in ('conda info --base 2^>nul') do (
      call :try_python "%%I\envs\stryke\python.exe"
      call :try_python "%%I\python.exe"
    )
  )
)

rem --- Fallback: py launcher (skips Windows Store alias stubs), then any python on PATH ---
if not defined PYTHON_EXE (
  for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do call :try_python "%%I"
)
if not defined PYTHON_EXE (
  for /f "delims=" %%I in ('where python 2^>nul') do call :try_python "%%I"
)
if not defined PYTHON_EXE (
  for /f "delims=" %%I in ('where py 2^>nul') do call :try_python "%%I"
)

if not defined PYTHON_EXE (
  echo [ERROR] Could not find a working Python executable for Stryke.
  echo         Set STRYKE_PYTHON_EXE to the full path of python.exe and run again.
  pause
  exit /b 1
)

set "FLASK_APP=webapp/app.py"
if not defined FLASK_SECRET_KEY set "FLASK_SECRET_KEY=dev-secret-key"
if not defined APP_PASSWORD set "APP_PASSWORD=expensive5rudabega!@1"
if not defined STRYKE_HOST set "STRYKE_HOST=127.0.0.1"
if not defined STRYKE_PORT set "STRYKE_PORT=5000"

echo Starting Stryke web app...
echo.
echo Open your browser to: http://%STRYKE_HOST%:%STRYKE_PORT%
echo Password: %APP_PASSWORD%
echo Using Python: %PYTHON_EXE%
echo.

"%PYTHON_EXE%" -c "import flask" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Missing dependency: flask
  echo         Install once with:
  echo         "%PYTHON_EXE%" -m pip install -r webapp\requirements.txt
  pause
  exit /b 1
)

start "" powershell -NoProfile -WindowStyle Hidden -Command "$u='http://%STRYKE_HOST%:%STRYKE_PORT%'; for($i=0;$i -lt 60;$i++){try{Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 1 ^> $null; Start-Process $u; break}catch{Start-Sleep -Milliseconds 500}}"

"%PYTHON_EXE%" -m flask run --host=%STRYKE_HOST% --port=%STRYKE_PORT%
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo [ERROR] Stryke web app exited with code %RC%.
  pause
)

exit /b %RC%

rem --- Accepts a candidate python.exe path; only keeps it if it actually runs ---
:try_python
if defined PYTHON_EXE goto :eof
if "%~1"=="" goto :eof
if not exist "%~1" goto :eof
"%~1" -c "import sys" >nul 2>nul
if not errorlevel 1 set "PYTHON_EXE=%~1"
goto :eof
