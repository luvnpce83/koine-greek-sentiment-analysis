#!/bin/bash
# 이 스크립트는 로컬 개발 환경을 자동으로 설정합니다.

echo "--- Cleaning up old virtual environment if it exists ---"
rm -rf venv

echo "--- Creating a new virtual environment named 'venv' ---"
python3 -m venv venv

echo "--- Upgrading pip in the new environment ---"
./venv/bin/pip install --upgrade pip

echo "--- Installing all required packages from requirements.txt ---"
./venv/bin/pip install -r requirements.txt

echo "✅ Local environment setup is complete!"
echo "To activate the new environment, please run the following command:"
echo "source venv/bin/activate"