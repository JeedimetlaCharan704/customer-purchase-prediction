@echo off
title Customer Purchase Prediction Dashboard
echo Starting Customer Purchase Prediction Backend...
cd /d "%~dp0"
start http://127.0.0.1:5001
python backend\app.py
pause
