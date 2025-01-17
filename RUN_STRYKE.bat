@echo off

:: Get the batch file path
set "BAT_PATH=%~dp0"

:: Go up two levels to the main directory
::for %%A in ("%BAT_PATH%..") do set "MAIN_DIR=%%~fA"

:: Construct paths relative to the base directory
set "STRYKE_PATH=%BAT_PATH%"
set "NOTEBOOK_PATH=%BAT_PATH%stryke_voila.ipynb"
set "SCRIPT_PATH=%BAT_PATH%Scripts\run_simulation_voila.py"

:: Print paths for verification (optional for troubleshooting)
::echo STRYKE_PATH: "%STRYKE_PATH%"
::echo NOTEBOOK_PATH: "%NOTEBOOK_PATH%"
::echo SCRIPT_PATH: "%SCRIPT_PATH%"

echo Launching Stryke . . .



:: Check for Anaconda installation in different locations
set "ACTIVATE_SCRIPT="
if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" (
    set "ACTIVATE_SCRIPT=C:\ProgramData\Anaconda3\Scripts\activate.bat"
) else if exist "C:\Users\%USER_NAME%\AppData\Local\anaconda3\Scripts\activate.bat" (
    set "ACTIVATE_SCRIPT=C:\Users\%USER_NAME%\AppData\Local\anaconda3\Scripts\activate.bat"
) else (
    echo Error: Could not find Anaconda installation.
    pause
    exit /b
)


:: Activate the Conda environment
call "%ACTIVATE_SCRIPT%"
call conda activate stryke


:: old ->
:: Activate the Conda environment
::call C:\ProgramData\Anaconda3\Scripts\activate.bat
::call conda activate stryke



:: Launch the Jupyter Notebook with Voila
call voila "%NOTEBOOK_PATH%"

pause



