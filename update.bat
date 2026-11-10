@echo off
rem Windows: refresh the data, rerun the model for my_team.txt and open the dashboard.
rem Double-click it, or run "update.bat auto" from Task Scheduler (no pause at the end).
rem Runs only on this computer. No server, no login, nothing uploaded.
cd /d "%~dp0"
set PYTHONUTF8=1
set "PY=py -3"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo Updating FPL model (live FPL data)...
%PY% fpl_model.py --team-file my_team.txt --refresh --open
if errorlevel 1 (
  echo.
  echo Live FPL site did not answer. Trying the GitHub copy of the data instead...
  %PY% fpl_model.py --team-file my_team.txt --source mirror --refresh --open
)
echo.
if /i not "%~1"=="auto" pause
