@echo off
echo ========================================================
echo   Setting up KnowLight Conda Environment
echo ========================================================

echo Cleaning corrupted conda package archives...
call conda clean --tarballs -y

echo Creating Conda environment 'knowlight'...
call conda env create -f "%~dp0environment.yml"
if errorlevel 1 (
    echo [ERROR] Conda environment creation failed!
    exit /b 1
)

echo.
echo ========================================================
echo   Activating environment 'knowlight'...
echo ========================================================
call conda activate knowlight

echo.
echo ========================================================
echo   Environment setup complete!
echo   To activate: conda activate knowlight
echo ========================================================
