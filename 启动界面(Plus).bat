@echo off
chcp 936 >nul
title 声音转换 Plus（关掉这个窗口就退出）
set PYTHONPATH=
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

rem 依次找可用的 python：优先本目录 .venv，其次 PATH 里的 python
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if not defined PY (
  where python >nul 2>&1 && set "PY=python"
)

if not defined PY (
  echo.
  echo   没找到 Python 运行环境。
  echo   第一步：双击「安装依赖.bat」会自动装；如果它提示没装 Python，
  echo           就去 python.org 下载安装，装的时候勾上 Add to PATH。
  echo.
  pause
  exit /b
)

echo.
echo   声音转换 Plus 正在启动，请稍等 1~2 秒……
echo   （黑窗口别关，关了程序就退出）
echo.
"%PY%" gui.py
echo.
echo 程序已退出。按任意键关闭窗口。
pause >nul
