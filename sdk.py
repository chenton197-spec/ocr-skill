"""
Qwen2-VL OCR Python SDK

用法:
    from sdk import OCR
    result = OCR("photo.png").extract()
    print(result)

    # 或批量
    results = OCR.batch(["a.png", "b.jpg"], lang="en")
"""
from pathlib import Path
from typing import Union, Optional

from extractor import TextExtractor


LANG_MAP = {
    "zh": "请用中文输出",
    "en": "Please output in English",
    "ja": "日本語で出力してください",
    "ko": "한국어로 출력해 주세요",
    "mixed": "Mixed languages — output as-is",
}

DEFAULT_PROMPT = "请提取图片中的所有文字内容，保持原有格式，如果包含表格请用 Markdown 表格格式输出。如果有公式请保留，有代码请用代码块包裹。"


class OCR:
    """Qwen2-VL OCR SDK"""

    _extractor: Optional[TextExtractor] = None

    def __init__(self, image_path: Union[str, Path], lang: str = "zh", prompt: Optional[str] = None):
        self.image_path = Path(image_path)
        self.lang = lang
        self.prompt = prompt or DEFAULT_PROMPT

    def extract(self) -> str:
        """提取图片文字"""
        if OCR._extractor is None:
            OCR._extractor = TextExtractor()

        lang_instruction = LANG_MAP.get(self.lang, "")
        full_prompt = f"{self.prompt} {lang_instruction}".strip()
        text, _ = OCR._extractor.extract(str(self.image_path), prompt=full_prompt)
        return text

    @classmethod
    def batch(cls, image_paths: list[Union[str, Path]], lang: str = "zh", prompt: Optional[str] = None):
        """批量提取多张图片文字"""
        return [cls(p, lang=lang, prompt=prompt).extract()[0] for p in image_paths]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python sdk.py <图片路径>")
        sys.exit(1)

    result = OCR(sys.argv[1]).extract()
    print(result)
