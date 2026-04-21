#!/bin/bash
# OCR Project 启动器 — 使用 conda robot 环境
export PATH="/home/casbot/miniconda3/envs/robot/bin:$PATH"
cd /home/casbot/workspace/ocr-project

# 取消代理（避免 httpx SOCKS 问题）
unset ALL_PROXY all_proxy

case "$1" in
  web)
    echo "🚀 启动 Gradio Web UI..."
    exec python app.py
    ;;
  api)
    echo "🚀 启动 FastAPI..."
    exec uvicorn api:app --host 0.0.0.0 --port 8000 --reload
    ;;
  download)
    echo "📦 下载模型..."
    exec python download_model.py
    ;;
  *)
    echo "用法: ./run.sh {web|api|download}"
    echo "  web  — Gradio Web UI (http://localhost:7860)"
    echo "  api  — FastAPI 服务  (http://localhost:8000)"
    echo "  download — 下载 Qwen2-VL 模型"
    ;;
esac
