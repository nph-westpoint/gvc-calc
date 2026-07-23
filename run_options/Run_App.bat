@echo off
echo ===================================================
echo Starting the GVC Calc App...
echo Please make sure Docker Desktop is running!
echo ===================================================
echo.

:: 1. Start or resume the container in detached mode (-d)
docker run -d -p 8501:8501 --name gvc-calc-app nphwestpoint/gvc-calc:1.1 2>nul || docker start gvc-calc-app

:: 2. Pause briefly to allow Streamlit/Python to initialize
timeout /t 3 /nobreak >nul

:: 3. Open the browser
start http://localhost:8501

echo.
echo Application started successfully at http://localhost:8501!
echo.
pause
