#!/bin/bash

echo "==================================================="
echo "Starting the GVC Calc App..."
echo "Please make sure Docker Desktop is running!"
echo "==================================================="
echo ""

# 1. Run new container or start existing one if it already exists
docker run -d -p 8501:8501 --name gvc-calc-app nphwestpoint/gvc-calc:1.1 2>/dev/null || docker start gvc-calc-app

# 2. Pause 3 seconds to let Streamlit boot up
sleep 3

# 3. Open the default web browser to the app
open http://localhost:8501

echo ""
echo "Application started successfully at http://localhost:8501!"
echo ""
