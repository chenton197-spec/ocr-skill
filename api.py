#!/usr/bin/env python3
"""
FastAPI REST API — Qwen2-VL 图片文字提取
启动: uvicorn api:app --reload --port 8000
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from extractor import TextExtractor

app = FastAPI(title="Qwen2-VL OCR API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局模型实例（启动时加载）
model: Optional[TextExtractor] = None


class OCRRequest(BaseModel):
    prompt: str = "请提取图片中的所有文字内容，保持原有格式。"
    language: str = "zh"
    max_tokens: int = 4096


class OCRResponse(BaseModel):
    text: str
    success: bool
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    global model
    print("🔄 加载 Qwen2-VL-2B-Instruct 模型...")
    model = TextExtractor()
    print("✅ 模型就绪")


@app.get("/")
async def root():
    return {"message": "Qwen2-VL OCR API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(...),
    prompt: str = Form("请提取图片中的所有文字内容，保持原有格式。"),
    language: str = Form("zh"),
):
    """上传图片，提取文字"""
    if model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")

    try:
        image_bytes = await file.read()
        lang_map = {"zh": "中文", "en": "英文", "ja": "日文", "ko": "韩文", "mixed": "混排"}
        lang_instruction = {
            "zh": "请用中文输出",
            "en": "Please output in English",
            "ja": "日本語で出力してください",
            "ko": "한국어로 출력해 주세요",
            "mixed": "Mixed languages — output as-is",
        }.get(language, "")

        full_prompt = f"{prompt} {lang_instruction}".strip()
        text, _ = model.extract(image_bytes, prompt=full_prompt)

        return OCRResponse(text=text, success=True)
    except Exception as e:
        return OCRResponse(text="", success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
