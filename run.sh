#!/bin/bash
# US Market Intelligence System - Launcher
# 미국 증시 인텔리전스 시스템 실행 스크립트

echo "======================================"
echo " US Market Intelligence System"
echo " 미국 증시 인텔리전스 시스템"
echo "======================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies (패키지 설치 중)..."
pip install -r requirements.txt -q

echo ""
echo "Starting application on http://localhost:8501"
echo "브라우저에서 http://localhost:8501 를 열어주세요"
echo ""
echo "Press Ctrl+C to stop | 종료하려면 Ctrl+C 를 누르세요"
echo "======================================"

streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    --theme.base dark \
    --theme.primaryColor "#FFA500" \
    --theme.backgroundColor "#0E1117" \
    --theme.secondaryBackgroundColor "#1E2130" \
    --theme.textColor "#FFFFFF"
