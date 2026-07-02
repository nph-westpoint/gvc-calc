@echo off
echo ==========================================
echo Starting the GVC Calc App...
echo Please make sure Docker Desktop is running!
echo ==========================================

start http://localhost:8501
docker run -p 8501:8501 nphwestpoint/gvc_calc:1.0
pause
