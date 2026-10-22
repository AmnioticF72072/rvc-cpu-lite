@echo off
chcp 936 >nul
title 安装运行依赖（第一次用，只装一次）
set PYTHONPATH=
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
  echo.
  echo   没检测到 Python。请先去 https://www.python.org/downloads/ 下载安装，
  echo   安装时一定勾选下面的 "Add python.exe to PATH"，装完再回来双击这个。
  echo.
  pause
  exit /b
)

echo 正在安装运行依赖（用国内镜像，约 2~5 分钟）……
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
echo.
echo 装好了。现在双击「启动界面(Plus).bat」就能用。
echo （以后不用再装，除非提示缺什么包。）
echo.
pause
