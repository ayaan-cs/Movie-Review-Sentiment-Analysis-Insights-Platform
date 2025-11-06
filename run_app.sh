#!/bin/bash
# Helper script to run Streamlit app with correct virtual environment

cd "$(dirname "$0")"
source venv/bin/activate
echo "Using Python: $(which python)"
echo "Starting Streamlit app..."
streamlit run app.py

