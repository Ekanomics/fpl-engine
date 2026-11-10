@echo off
rem Show one player's stats and projections in this window:  player.bat Haaland
cd /d "%~dp0"
set PYTHONUTF8=1
if "%~1"=="" (echo Usage: player.bat NAME   e.g. player.bat Haaland & exit /b 1)
.venv\Scripts\python.exe fpl_model.py --team-file my_team.txt --player "%~1" --no-dashboard --no-backtest
