#!/home/casbot/miniconda3/envs/robot/bin/python3
"""
Qwen2-VL OCR 命令行工具

用法:
    python ocr.py <图片路径>                    # 提取文字
    python ocr.py <图片路径> -o output.txt       # 保存到文件
    python ocr.py <图片路径> --prompt "自定义指令"
    python ocr.py <图片路径> --lang en           # 英文
    python ocr.py <图片1.png> <图片2.jpg> ...    # 批量提取

示例:
    python ocr.py screenshot.png
    python ocr.py doc.pdf --lang zh -o result.txt
"""
import argparse
import sys
from pathlib import Path

# 确保能找到 extractor
sys.path.insert(0, str(Path(__file__).parent))

from extractor import TextExtractor


LANG_MAP = {
    "zh": "请用中文输出",
    "en": "Please output in English",
    "ja": "日本語で出力してください",
    "ko": "한국어로 출력해 주세요",
    "mixed": "Mixed languages — output as-is",
}

DEFAULT_PROMPT = "列出图片中所有可见的文字内容，直接输出文字，不要输出坐标、边框或位置信息。"


def main():
    parser = argparse.ArgumentParser(
        description="Qwen2-VL 图片文字提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("images", nargs="+", help="图片路径（支持多张）")
    parser.add_argument("-o", "--output", help="输出文件路径（默认输出到stdout）")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT, help="提取指令")
    parser.add_argument("--lang", default="zh", choices=LANG_MAP.keys(), help="输出语言（默认中文）")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式，不打印状态信息")

    args = parser.parse_args()

    extractor = TextExtractor()
    results = []

    for img_path in args.images:
        path = Path(img_path)
        if not path.exists():
            print(f"❌ 文件不存在: {img_path}", file=sys.stderr)
            continue

        if not args.quiet:
            print(f"📄 正在处理: {img_path}")

        lang_instruction = LANG_MAP[args.lang]
        full_prompt = f"{args.prompt} {lang_instruction}".strip()

        try:
            text, inference_time = extractor.extract(str(path), prompt=full_prompt)
            results.append(f"# {path.name}\n{text}")
            if not args.quiet:
                print(f"✅ 提取完成 ({len(text)} 字符)，推理耗时: {inference_time:.2f}s")
        except Exception as e:
            print(f"❌ 处理失败: {img_path} — {e}", file=sys.stderr)
            results.append(f"# {path.name}\n[提取失败: {e}]")

    output = "\n\n".join(results)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        if not args.quiet:
            print(f"💾 已保存到: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
