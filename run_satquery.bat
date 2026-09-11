@echo off
echo ===================================================
echo Starting SatQuery AI Backend and Web GUI...
echo ===================================================
cd /d "%~dp0"
start "" "http://127.0.0.1:8000"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
pause
