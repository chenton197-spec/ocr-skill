# ocr-skill

基于 **Qwen2-VL-2B-Instruct** 的图片文字识别工具。

## 触发条件

用户要求从图片中提取文字时使用，包括：截图 OCR、文档识别、表格提取、公式识别等场景。

## 使用方式

### 1. 命令行（单次任务）

```bash
cd /home/casbot/workspace/ocr-skill
./ocr.py <图片路径> [-o <输出文件>] [--prompt "<自定义指令>"] [--lang zh|en|ja|ko|mixed]
```

**示例：**
```bash
./ocr.py /home/casbot/图片/test.png
./ocr.py screenshot.png -o result.txt --lang en
./ocr.py doc.png --prompt "请提取表格内容"
```

### 2. FastAPI 服务（推荐，模型常驻，响应更快）

```bash
cd /home/casbot/workspace/ocr-skill
./run.sh api
# 启动后模型常驻显存，后续调用无需重新加载
```

**调用：**
```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@<图片路径>" \
  -F "prompt=列出图片中所有可见的文字内容" \
  -F "language=zh"
```

### 3. Python 模块

```python
from extractor import TextExtractor

extractor = TextExtractor()
text, inference_time = extractor.extract("<图片路径或URL或bytes>")
print(text)
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | 列出图片中所有可见的文字内容 | 给模型的指令，越具体结果越准 |
| `language` | `zh` | 输出语言：zh / en / ja / ko / mixed |
| `min_pixels` | 256×28² | 越小越快但可能丢细节 |
| `max_tokens` | 512 | 最大输出 token 数 |

## 性能基准

| 场景 | 推理时间 | 说明 |
|------|---------|------|
| CLI 单次（不含模型加载） | ~1.5s | 每次重新加载模型（40s） |
| API 服务（模型常驻） | ~1.5s | 无加载开销 |

> **提示：** 频繁调用建议启动 API 服务（`./run.sh api`），避免每次加载模型。

## 注意事项

- **环境要求：** CUDA GPU（4GB+ 显存），Python 环境需在 `/home/casbot/miniconda3/envs/robot/bin/python3`
- **路径问题：** 直接 `python ocr.py` 会缺 torch，应用 `./ocr.py` 或指定 conda python
- **模型路径：** `/home/casbot/workspace/ocr-skill/models/Qwen2-VL-2B-Instruct`
- **输出稳定性：** 同一图片多次提取可能有微小差异（大模型固有特性）

## 故障排除

| 问题 | 解决 |
|------|------|
| `No module named 'torch'` | 确认使用 conda robot 环境：`./ocr.py` 或 `/home/casbot/miniconda3/envs/robot/bin/python3 ocr.py` |
| API 返回 `tokens: 0, features: 44` | 检查图片格式是否损坏，尝试重新保存 |
| 识别结果有坐标非文字 | 换用明确 prompt：`--prompt "列出图片中所有可见的文字内容，直接输出文字，不要输出坐标"` |
