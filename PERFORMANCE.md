# Qwen2-VL OCR 推理速度优化

## 当前性能基准

| 图片 | 分辨率 | vision tokens | 推理时间 | 硬件 |
|------|--------|---------------|---------|------|
| test.png | 148×101 | ~1160 | ~0.6s | RTX 5060 (8GB) |
| 1.png | 310×100 | ~1160 | ~1.5s | RTX 5060 (8GB) |

> 注：推理时间不含模型加载（约 40s），适合 API 服务常驻内存后调用。

---

## 已应用的优化

### 1. 修复重复 generate Bug（最大加速 ~2x）
**文件**: `extractor.py`

`generate()` 被调用了两次，第一次结果直接丢弃。
```python
# ❌ 修复前 — generate 跑了两次
output_ids = self._model.generate(**inputs, ...)
t0 = time.perf_counter()
generated_ids = self._model.generate(**inputs, ...)  # 这次才被用！
t1 = time.perf_counter()

# ✅ 修复后 — 只跑一次
t0 = time.perf_counter()
with torch.inference_mode():
    generated_ids = self._model.generate(**inputs, ...)
t1 = time.perf_counter()
```

### 2. cudnn.benchmark（GPU 卷积自动选最优 kernel）
```python
torch.backends.cudnn.benchmark = True
```

### 3. Flash Attention / SDPA（减少显存访问）
```python
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
```

### 4. torch.compile — JIT 编译（提速 10~30%）
```python
self._model = torch.compile(self._model, mode="max-autotune", fullgraph=False)
# 首次调用会触发编译（慢），之后调用显著加速
```

### 5. torch.inference_mode()（减少 autograd 开销）
```python
with torch.inference_mode():
    generated_ids = self._model.generate(**inputs, ...)
```

### 6. max_tokens 从 4096 降到 512（减少 LLM 生成长度）
OCR 输出通常很短，512 token 完全够用。

### 7. 修复默认 Prompt（避免误触发 bounding box 模式）
```python
# ❌ 修复前 — 某些图片触发坐标输出
"请提取图片中的所有文字内容，保持原有格式..."

# ✅ 修复后 — 明确要求只输出文字
"列出图片中所有可见的文字内容，直接输出文字，不要输出坐标、边框或位置信息。"
```

---

## 关键发现：Vision Token 数量是瓶颈

Qwen2-VL 的推理时间 ≈ vision encoder 处理时间（不是 LLM）。

| min_pixels | 等效像素 | vision tokens | 推理时间 | 精度 |
|-----------|---------|---------------|---------|------|
| 28²×4 = 15,680 | 4px | 176 | ~800ms | ⚠️ 丢行 |
| 28²×16 = 200,704 (默认) | 16px | ~1160 | ~1.5s | ✅ 完整 |
| 28²×36 = 282,24 | 36px | ~1500+ | ⚠️ OOM |

**发现**：
- vision tokens 越多越慢，但精度越好
- 中间值（如 min=28²×8）触发 CPU offload，反而更慢
- 小分辨率丢失"上层/下层"等细粒度换行信息

**建议**：
- 纯文字截图：`min_pixels=28²×16`（当前默认值），速度质量平衡
- 需要更快速度且接受精度损失：`min_pixels=28²×4`（约 800ms）
- 避免使用中间值

---

## 进一步提速建议

### 短期（改动小）
1. **API 服务模式** — 模型常驻内存，消除 40s 加载时间
   ```bash
   ./run.sh api
   # POST http://localhost:8000/extract
   ```
2. **AWQ 量化** — 4bit 量化可将模型体积从 4GB 压到 1GB，减少内存带宽压力（需要 bitsandbytes >= 0.46.1）

### 中期（需评估）
3. **换用专用 OCR 模型** — Qwen2-VL 是通用 VLM，专用 OCR 模型（如 PaddleOCR EasyOCR）在纯文字提取场景快 10~50 倍
4. **TensorRT 部署** — 将模型导出为 TensorRT 格式，推理可提速 2~5x

### 长期
5. **升级 GPU** — 更大显存允许更大 batch 和更少 CPU offload
6. **换更大模型** — Qwen2.5-VL 系列对视觉任务有更好的视觉编码器

---

## 环境要求

```bash
# 推荐 conda 环境
/home/casbot/miniconda3/envs/robot/bin/python3

# 依赖
pip install torch transformers pillow
# 可选（进一步量化加速）
pip install bitsandbytes>=0.46.1
```
