"""Gradio receipt uploader with honest baseline behaviour.

No fake side-by-side output: baseline requires either a verified adapter-disable
context on the loaded model or a previously recorded exact-image prediction.
"""
from pathlib import Path
import hashlib
import io
import json

from .extract import parse_receipt
from .vision import predict_receipt


def image_digest(image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def make_demo(model, processor, config, baseline_cache: dict | None = None):
    import gradio as gr
    baseline_cache = baseline_cache or {}

    def do_extract(image):
        if image is None:
            raise gr.Error("Upload a receipt image first")
        fine_text = predict_receipt(model, processor, image,
                                    max_edge=config.max_image_edge,
                                    max_new_tokens=config.max_new_tokens)
        parsed = parse_receipt(fine_text)
        base_text = "Baseline unavailable for this image; original-model predictions are stored in the evaluation CSV."
        # PEFT may offer an adapter-disable context for exact same loaded model.
        # Never silently claim an adapted prediction is an original prediction.
        disable_adapter = getattr(model, "disable_adapter", None)
        if callable(disable_adapter):
            try:
                with disable_adapter():
                    base_text = predict_receipt(model, processor, image,
                        max_edge=config.max_image_edge,
                        max_new_tokens=config.max_new_tokens)
            except Exception as exc:
                base_text = f"Baseline could not run live ({type(exc).__name__}). Use saved evaluation results."
        return base_text, fine_text, parsed or {"error": "Output is not valid four-field JSON", "raw": fine_text}

    with gr.Blocks(title="ReceiptLens — Gemma 4 Receipt Extraction") as app:
        gr.Markdown("# ReceiptLens\nFine-tuned Gemma 4 E2B extracts receipt fields. The plain model is shown live only when adapter disabling works; otherwise review the saved baseline CSV.")
        with gr.Row():
            image = gr.Image(type="pil", label="Upload receipt image")
            with gr.Column():
                button = gr.Button("Extract", variant="primary")
                json_out = gr.JSON(label="Parsed fine-tuned result")
        with gr.Row():
            plain = gr.Textbox(label="Plain Gemma 4 (if available)", lines=10)
            tuned = gr.Textbox(label="LoRA fine-tuned Gemma 4", lines=10)
        button.click(do_extract, inputs=image, outputs=[plain, tuned, json_out])
    return app
