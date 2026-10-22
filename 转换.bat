@echo off
chcp 936 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if exist .venv\Scripts\python.exe (
  set PY=.venv\Scripts\python.exe
) else (
  set PY=python
)
if "%~1"=="" (
  "%PY%" convert.py
) else (
  "%PY%" convert.py -i "%~1"
)
pause
