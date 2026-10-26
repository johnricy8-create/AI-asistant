@echo off
cd /d %~dp0
echo ============================================
echo  优选商城 - 智能客服 Copilot Demo
echo  http://127.0.0.1:8000
echo ============================================

if exist "D:\pytorch\.venv\Scripts\python.exe" (
  echo 使用 D:\pytorch\.venv（ChromaDB + SiliconFlow）
  "D:\pytorch\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
) else (
  echo 未找到 D:\pytorch\.venv，回退到项目 .venv（词法检索）
  uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
)
pause
