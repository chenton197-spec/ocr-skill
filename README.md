# Qwen2-VL OCR — Image Text Extraction Tool

> Extract text from any image with a single command or API call. Powered by Qwen2-VL-2B-Instruct vision language model.

## What is this?

A production-ready **image-to-text (OCR)** tool built on **Qwen2-VL-2B-Instruct**, capable of recognizing text from screenshots, documents, receipts, business cards, tables, and complex layouts — with support for **Chinese, English, Japanese, Korean**, and mixed-language content.

Unlike traditional OCR engines (Tesseract, PaddleOCR), this tool uses a **vision-language model** to understand context, making it significantly more robust on ambiguous or unstructured images.

## Features

- 🖼️ **Any image type** — screenshots, photos, scanned documents, PDFs, tables, handwritten notes
- 🌏 **Multilingual** — Chinese (中文), English, Japanese (日本語), Korean (한국어), mixed
- 📊 **Table & formula aware** — preserves table structures, handles mathematical expressions
- ⚡ **GPU accelerated** — CUDA-optimized with BF16, ~1.5s per image on RTX 5060
- 🚀 **Three interfaces** — CLI, REST API, Python SDK (drop-in module)
- 🔧 **Configurable** — adjustable resolution, prompt, output length

## Model Download

The model (`Qwen2-VL-2B-Instruct`, ~4 GB) must be downloaded before first use and placed under `models/Qwen2-VL-2B-Instruct/`.

### Method 1 — Built-in script (recommended)

Downloads directly from HuggingFace with **resume support** (safe to interrupt and retry):

```bash
./run.sh download
# or equivalently
python download_model.py
```

Files saved to `models/Qwen2-VL-2B-Instruct/`:
| File | Size |
|------|------|
| `model-00001-of-00002.safetensors` | ~3.6 GB |
| `model-00002-of-00002.safetensors` | ~410 MB |

> If behind a proxy, **unset** SOCKS proxies first — `unset ALL_PROXY all_proxy` — or the download will fail.

### Method 2 — huggingface_hub

```bash
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    local_dir="models/Qwen2-VL-2B-Instruct",
    ignore_patterns=["*.bin"],   # skip redundant .bin weights
)
EOF
```

### Method 3 — ModelScope (国内推荐，无需代理)

```bash
pip install modelscope
python - <<'EOF'
from modelscope import snapshot_download
snapshot_download(
    model_id="qwen/Qwen2-VL-2B-Instruct",
    cache_dir="models",
)
EOF
# ModelScope 保存路径与 HuggingFace 不同，需手动重命名或修改 extractor.py 中的 MODEL_DIR
```

### Method 4 — git-lfs (manual)

```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct \
          models/Qwen2-VL-2B-Instruct
```

### Verify download

```bash
ls -lh models/Qwen2-VL-2B-Instruct/*.safetensors
# Expected: two files totalling ~4 GB
```

---

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url>
cd ocr-skill
pip install -r requirements.txt

# 2. Download model (see "Model Download" section above)
./run.sh download

# 3. Run CLI
./ocr.py screenshot.png

# Or start API server (recommended for frequent use)
./run.sh api
curl -X POST http://localhost:8000/ocr \
  -F "file=@screenshot.png" \
  -F "prompt=列出图片中所有可见的文字内容" \
  -F "language=zh"
```

## Use Cases

| Scenario | Command |
|----------|---------|
| Screenshot OCR | `./ocr.py screenshot.png` |
| Chinese document | `./ocr.py doc.jpg --lang zh` |
| English receipt | `./ocr.py receipt.png --lang en` |
| Table extraction | `./ocr.py table.png --prompt "请提取表格内容，用 Markdown 表格输出"` |
| Batch processing | `./ocr.py img1.png img2.jpg img3.pdf` |
| Save to file | `./ocr.py image.png -o output.txt` |

## Performance

| Setup | Inference Time | Notes |
|-------|---------------|-------|
| CLI (per call) | ~1.5s | Reloads model every call |
| API server | ~1.5s | Model stays in GPU memory |

**Hardware:** NVIDIA RTX 5060 8GB, Qwen2-VL-2B-Instruct (FP16, ~4GB model)

## Python SDK

```python
from extractor import TextExtractor

extractor = TextExtractor()
text, elapsed = extractor.extract("image.png")
print(text)  # "10:00-11:30 上层\n13:00-15:00 下层..."
```

## Architecture

```
ocr-skill/
├── extractor.py      # TextExtractor — core inference class
├── api.py            # FastAPI REST server (port 8000)
├── app.py            # Gradio Web UI (port 7860)
├── ocr.py           # CLI entry point
├── run.sh           # Start script (web | api | download)
└── models/          # Qwen2-VL-2B-Instruct weights
```

## Comparison with Alternatives

| Tool | Model Type | Chinese Accuracy | Speed (GPU) | Setup Complexity |
|------|-----------|-----------------|-------------|-----------------|
| **This project** | VLM (Qwen2-VL-2B) | ✅ Excellent | ~1.5s | Low |
| PaddleOCR | CNN+CRNN | ✅ Good | ~0.1s | Medium |
| Tesseract | LSTM | ⚠️ Average | ~0.5s | Low |
| EasyOCR | CNN+Transformer | ✅ Good | ~0.3s | Low |

**When to choose this:** Need high accuracy on complex/unstructured images, especially with Chinese text or mixed languages, and can spare a GPU.

## Environment

- Python: `/home/casbot/miniconda3/envs/robot/bin/python3`
- CUDA: 12.x+
- GPU memory: 4GB+ (8GB recommended)
- OS: Linux

## Keywords (for AI search)

`OCR` `image to text` `text extraction` `文字识别` `光学字符识别` `Qwen2-VL` `vision language model` `VLM` `Chinese OCR` `screenshot OCR` `table recognition` `formula recognition` `math OCR` `multilingual OCR` `PaddleOCR alternative` `transformers` `HuggingFace` `Python OCR` `screenshot text extraction` `document OCR`
# ocr-skill
