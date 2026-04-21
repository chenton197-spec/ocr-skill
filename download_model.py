#!/usr/bin/env python3
"""
可靠的模型下载脚本 — 使用 requests Session 断点续传
"""
import os
import sys
import time
import requests
from pathlib import Path

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
LOCAL_DIR = Path(__file__).parent / "models" / "Qwen2-VL-2B-Instruct"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# 文件列表及大小（MB）
FILES = [
    ("model-00001-of-00002.safetensors", 3628),
    ("model-00002-of-00002.safetensors", 410),
]

BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main"

def get_url(filename):
    return f"{BASE_URL}/{filename}"

def download_file(filename, expected_mb, retries=5):
    path = LOCAL_DIR / filename
    url = get_url(filename)
    resume_header = {}

    if path.exists() and path.stat().st_size > 1024:
        print(f"  📂 已存在: {filename} ({path.stat().st_size/1024/1024:.1f} MB)，跳过")
        return True

    incomplete = LOCAL_DIR / ".cache" / "huggingface" / "download" / f"{filename}.incomplete"
    if incomplete.exists():
        resume_size = incomplete.stat().st_size
        if resume_size > 1024:
            resume_header["Range"] = f"bytes={resume_size}-"
            print(f"  ⏳ 断点续传: {filename} 从 {resume_size/1024/1024:.1f} MB 继续")
            path = incomplete

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; Qwen2VL-Downloader/1.0)",
    })

    for attempt in range(retries):
        try:
            r = session.get(url, headers=resume_header, stream=True, timeout=60)
            r.raise_for_status()

            total = int(r.headers.get("Content-Length", 0))
            downloaded = resume_header.get("Range", "") and int(resume_header["Range"].split("=")[1].split("-")[0]) or 0

            print(f"  ⬇️  下载中: {filename} ({expected_mb} MB)")
            with open(path, "ab" if resume_header else "wb") as f:
                for chunk in r.iter_content(chunk_size=1 * 1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        pct = min(100, downloaded / total * 100) if total else 0
                        print(f"\r  ⏬ {downloaded/1024/1024:.1f}/{expected_mb} MB ({pct:.0f}%)", end="", flush=True)
            print()  # newline after progress

            if incomplete.exists():
                incomplete.rename(LOCAL_DIR / filename)
            print(f"  ✅ 完成: {filename}")
            return True

        except Exception as e:
            print(f"  ⚠️  错误 (尝试 {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
            resume_header = {}  # 重试时从头开始

    print(f"  ❌ 放弃: {filename}")
    return False

def main():
    print(f"📦 下载模型: {MODEL_ID}")
    print(f"📁 保存到: {LOCAL_DIR}\n")

    success = 0
    for filename, expected_mb in FILES:
        if download_file(filename, expected_mb):
            success += 1
        print()

    print(f"\n🎉 完成 ({success}/{len(FILES)} 文件)")
    if success == len(FILES):
        print("✅ 可以运行 python app.py 启动服务")

if __name__ == "__main__":
    main()
