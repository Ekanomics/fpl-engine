@echo off
rem Windows one-time setup: private Python environment + pinned libraries + your team file.
rem Needs Python 3.11+ from python.org or "winget install Python.Python.3.12".
cd /d "%~dp0"
where py >nul 2>nul || (echo Python launcher "py" not found. Install Python 3.12 first, then run this again. & pause & exit /b 1)
py -3 --version
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" || (echo Python 3.11 or newer is needed. & pause & exit /b 1)
if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || (echo Library install failed. & pause & exit /b 1)
if not exist "my_team.txt" copy my_team.example.txt my_team.txt >nul
echo.
echo Setup complete. Edit my_team.txt, then double-click update.bat.
pause
