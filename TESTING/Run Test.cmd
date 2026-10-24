@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0run_test.py"
) else if exist "%~dp0..\.venv\Scripts\python.exe" (
  "%~dp0..\.venv\Scripts\python.exe" "%~dp0run_test.py"
) else (
  python "%~dp0run_test.py"
)
if errorlevel 1 echo Test did not complete. Read the error above and Test Set ReadMe.md.
pause
