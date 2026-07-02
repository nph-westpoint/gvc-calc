#!/bin/bash
echo "=========================================="
echo "Starting the GVC Calc App..."
echo "Please make sure Docker Desktop is running!"
echo "=========================================="

(sleep 3 && open http://localhost:8501) &
docker run -p 8501:8501 nphwestpoint/gvc_calc:1.0
