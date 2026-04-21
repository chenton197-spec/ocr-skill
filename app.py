#!/usr/bin/env python3
"""
Gradio Web UI — Qwen2-VL 图片文字提取
启动: python app.py
"""
import sys
import gradio as gr
import PIL.Image
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from extractor import TextExtractor

__version__ = "1.0.0"

DEFAULT_PROMPT = "请提取图片中的所有文字内容，保持原有格式。如果包含表格请用 Markdown 表格输出，如果有公式请保留。"

model = None

def get_model():
    global model
    if model is None:
        model = TextExtractor()
    return model

def extract_from_image(image, prompt=DEFAULT_PROMPT, language="中文"):
    if image is None:
        return "", "⚠️ 请上传图片"

    lang_instruction = {
        "中文": "请用中文输出",
        "英文": "Please output in English",
        "日文": "日本語で出力してください",
        "韩文": "한국어로 출력해 주세요",
        "混排": "Mixed languages — output as-is",
    }.get(language, "")

    full_prompt = f"{prompt} {lang_instruction}".strip()

    try:
        extractor = get_model()
        result = extractor.extract(image, prompt=full_prompt)
        return result, "✅ 提取完成"
    except Exception as e:
        return "", f"❌ 错误: {str(e)}"

def main():
    import gradio as gr

    with gr.Blocks(
        title="📸 Qwen2-VL 文字提取",

    ) as demo:
        gr.Markdown("""
        # 📸 Qwen2-VL 图片文字提取
        基于 **Qwen2-VL-2B-Instruct** 模型，支持中英文、表格、公式、截图等。
        """)
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 上传图片",
                    type="pil",
                    height=400,
                )
                prompt_input = gr.Textbox(
                    label="📝 提取指令",
                    value=DEFAULT_PROMPT,
                    lines=3,
                )
                language = gr.Dropdown(
                    label="🌐 输出语言",
                    choices=["中文", "英文", "日文", "韩文", "混排"],
                    value="中文",
                )
                extract_btn = gr.Button("🔍 开始提取", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="📄 提取结果",
                    lines=20,
                    show_copy_button=True,
                )
                status = gr.Label(label="状态")

        gr.Examples(
            examples=[],  # 用户可自行拖入图片测试
            inputs=[image_input],
        )

        extract_btn.click(
            fn=extract_from_image,
            inputs=[image_input, prompt_input, language],
            outputs=[output_text, status],
        )

        gr.Markdown(f"""
        ---
        **提示：** 可以上传截图、文档照片、PDF 截图等任意包含文字的图片。
        模型支持自然语言理解，能自动识别表格、公式、特殊符号等复杂布局。
        """)

    print("🚀 启动 Gradio UI: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

if __name__ == "__main__":
    main()
