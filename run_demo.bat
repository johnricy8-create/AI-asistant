@echo off
cd /d %~dp0
echo ============================================
echo  优选商城 - 智能客服 Copilot Demo
echo  http://127.0.0.1:8000
echo ============================================
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
