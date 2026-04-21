#!/usr/bin/env python3
"""
Qwen2-VL 图片文字提取核心模块
"""
import io
import time
from pathlib import Path
from typing import Union, Optional

import torch  # noqa: E402
from PIL import Image  # noqa: E402


MODEL_DIR = Path(__file__).parent / "models" / "Qwen2-VL-2B-Instruct"
DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"


class TextExtractor:
    """使用 Qwen2-VL 从图片中提取文字"""

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        compiled: bool = True,
    ):
        """
        Args:
            model_id: HuggingFace 模型 ID 或本地路径
            device: 设备，"cuda" 或 "cpu"
            min_pixels: Qwen2VL 最小处理像素数（必须是 28 的倍数），越小越快但可能损失细节
            max_pixels: Qwen2VL 最大处理像素数（必须是 28 的倍数）
            compiled: 是否使用 torch.compile 加速（首次调用会触发 JIT 编译，稍慢）
        """
        self.model_id = model_id or DEFAULT_MODEL
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.compiled = compiled
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

        model_path = str(MODEL_DIR) if MODEL_DIR.exists() else self.model_id

        # GPU 推理优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)

        print(f"🔄 加载模型: {model_path}")
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            trust_remote_code=True,
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # torch.compile JIT 编译（仅在 API 常驻服务时有收益，CLI 每次重载反而增加开销）
        # if self.compiled:
        #     self._model = torch.compile(self._model, mode="reduce-overhead", fullgraph=False)

        print("✅ 模型加载完成")

    def _encode_image(self, image_source: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        if isinstance(image_source, (str, Path)):
            path = Path(image_source)
            if path.exists():
                return Image.open(path).convert("RGB")
            import urllib.request
            with urllib.request.urlopen(str(image_source), timeout=30) as resp:
                data = resp.read()
            return Image.open(io.BytesIO(data)).convert("RGB")
        elif isinstance(image_source, bytes):
            return Image.open(io.BytesIO(image_source)).convert("RGB")
        elif isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        raise ValueError(f"Unsupported image source type: {type(image_source)}")

    def extract(
        self,
        image_source: Union[str, Path, bytes, Image.Image],
        prompt: str = "列出图片中所有可见的文字内容，直接输出文字，不要输出坐标、边框或位置信息。",
        max_tokens: int = 512,
    ) -> tuple[str, float]:
        """
        从图片中提取文字

        Args:
            image_source: 图片路径/URL/bytes/PIL Image
            prompt: 给模型的指令
            max_tokens: 最大生成的 token 数（OCR 通常 512 足够）

        Returns:
            提取出的文字内容 (str, 推理耗时秒数)
        """
        self._load_model()

        image = self._encode_image(image_source)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        t1 = time.perf_counter()

        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        result = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return result, (t1 - t0)

    def extract_batch(
        self,
        image_sources: list,
        prompt: str = "列出图片中所有可见的文字内容，直接输出文字。",
        max_tokens: int = 512,
    ) -> tuple[list[str], float]:
        """批量提取多张图片的文字"""
        self._load_model()

        images = [self._encode_image(src) for src in image_sources]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
            for img in images
        ]

        texts = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        t1 = time.perf_counter()

        results = []
        for i, ids in enumerate(output_ids):
            input_len = inputs["input_ids"][i].shape[0]
            gen_ids = ids[input_len:]
            text = self._processor.decode(gen_ids, skip_special_tokens=True)
            results.append(text)

        return results, (t1 - t0)


# 单图提取的快捷函数
_default_extractor: Optional[TextExtractor] = None


def extract_text(
    image_source: Union[str, Path, bytes, Image.Image],
    prompt: str = "列出图片中所有可见的文字内容，直接输出文字，不要输出坐标、边框或位置信息。",
    max_tokens: int = 512,
) -> tuple[str, float]:
    """快捷函数：单图文字提取"""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = TextExtractor()
    return _default_extractor.extract(image_source, prompt, max_tokens)


if __name__ == "__main__":
    print("✅ TextExtractor 模块正常，请通过 app.py 或 api.py 使用")
