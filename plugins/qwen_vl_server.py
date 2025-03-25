from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import List, Optional
import uvicorn

app = FastAPI()

# 初始化模型和处理器
@app.on_event("startup")
async def load_model():
    global model, processor
    
    # MPS设备初始化
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "or you do not have an MPS-enabled device on this machine.")
    else:
        pass  # MPS自动管理内存

    # 加载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "/Users/xishi/Documents/comfy/ComfyUI/models/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float32,  # MPS目前只支持float32
        device_map="mps",
    )

    # 优化processor配置
    min_pixels = 256*28*28
    max_pixels = 1024*28*28
    processor = AutoProcessor.from_pretrained(
        "/Users/xishi/Documents/comfy/ComfyUI/models/Qwen2.5-VL-7B-Instruct",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        use_fast=True  # 启用快速图像处理器
    )

class ImageRequest(BaseModel):
    image_path: str
    prompt: str

@app.post("/generate")
async def generate_description(request: ImageRequest):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": request.image_path,
                    },
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]

        # 准备推理
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("mps")

        # 推理
        generated_ids = model.generate(**inputs, max_new_tokens=1536)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return {"result": output_text[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)