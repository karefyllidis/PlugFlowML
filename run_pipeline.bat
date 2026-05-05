@echo off
REM HydrAI: run ML notebooks in order (Windows). Requires Python on PATH.
setlocal
cd /d "%~dp0"
python run_pipeline.py
if errorlevel 1 exit /b 1
