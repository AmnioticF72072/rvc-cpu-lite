@echo off
chcp 936 >nul
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set PYTHONIOENCODING=utf-8
rem 优先用包里的 runtime，这样连 Python 都不用装
set "PY=python"
if exist "runtime\python.exe" (
  set "PY=runtime\python.exe"
) else (
  if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
  )
)
if "%~1"=="" (
  "%PY%" convert.py
) else (
  "%PY%" convert.py -i "%~1"
)
pause
