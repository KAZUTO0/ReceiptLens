"""Gemma 4 vision wrapper. Heavy imports only when running in CUDA/Colab."""
from pathlib import Path
from .data import prepare_image
from .prompts import RECEIPT_INSTRUCTION


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Select a Colab T4 GPU runtime.")
    return torch


def load_model(model_id: str = "unsloth/gemma-4-E2B-it"):
    require_cuda()
    from unsloth import FastVisionModel, get_chat_template
    model, processor = FastVisionModel.from_pretrained(
        model_id, load_in_4bit=True, use_gradient_checkpointing="unsloth"
    )
    processor = get_chat_template(processor, "gemma-4")
    return model, processor


def predict_receipt(model, processor, image, *, max_edge=960, max_new_tokens=256):
    """Identical deterministic inference settings for base and adapted model."""
    torch = require_cuda()
    image = prepare_image(image, max_edge)
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": RECEIPT_INSTRUCTION}
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    model.eval()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=False, use_cache=True)
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    decoder = getattr(processor, "tokenizer", processor)
    return decoder.decode(generated_ids, skip_special_tokens=True).strip()


def add_lora(model, *, rank=8, alpha=8):
    from unsloth import FastVisionModel
    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=rank, lora_alpha=alpha, lora_dropout=0,
        bias="none", random_state=42,
    )


def save_adapter(model, processor, directory: str = "outputs/receipt_gemma4_lora"):
    Path(directory).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    return directory
