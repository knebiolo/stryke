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
:: This .bat file activates the Anaconda command prompt, then launches the stryke
:: environemnt and notebook. If you are seeing errors like 'Count not find Anaconda
:: installation, you will need to add the path to the Anaconda installation.
set "ACTIVATE_SCRIPT="
if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" (
    set "ACTIVATE_SCRIPT=C:\ProgramData\Anaconda3\Scripts\activate.bat"
) else if exist "C:\Users\%USER_NAME%\AppData\Local\anaconda3\Scripts\activate.bat" (
    set "ACTIVATE_SCRIPT=C:\Users\%USER_NAME%\AppData\Local\anaconda3\Scripts\activate.bat"
) else if exist "C:\%USERPROFILE%\Anaconda3\Scripts\activate.bat" (
    set ACTIVATE_SCRIPT=C:\%USERPROFILE%\Anaconda3\Scripts\activate.bat
) else if exist "C:\%USERPROFILE%\Anaconda\Scripts\activate.bat" (
    set ACTIVATE_SCRIPT=C:\%USERPROFILE%\Anaconda\Scripts\activate.bat
) else if exist "C:\%USERPROFILE%\Miniconda3\Scripts\activate.bat" (
    set ACTIVATE_SCRIPT=C:\%USERPROFILE%\Miniconda3\Scripts\activate.bat
) else (
    echo Error: Could not find Anaconda installation.
    pause
    exit /b
)


:: Activate the Conda environment
call "%ACTIVATE_SCRIPT%"
:: Get the list of all environments, pick out the one with 'stryke' in the name
for /f "tokens=1 delims= " %%E in ('conda env list ^| findstr /i "stryke"') do (
    call conda activate %%E 2>nul
    if %errorlevel% == 0 (
        echo Activated environment: %%E
        goto env_activated
    )
)
echo No specified Conda environment found. Exiting.
exit /b 1
:env_activated
echo Environment successfully activated.


:: Launch the Jupyter Notebook with Voila
call voila "%NOTEBOOK_PATH%"

pause



