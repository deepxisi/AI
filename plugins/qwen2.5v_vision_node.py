import comfy
import folder_paths
import torch
import requests
import json
import time
from PIL import Image
import io
import numpy as np
import os
import threading
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import uvicorn


class Qwen2_5vModelManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        torch.mps.empty_cache()
        self.model = None
        self.processor = None
        self._load_model()
        
    def _load_model(self):
        if self.model is not None:
            return
            
        print("⏳ 正在加载Qwen2.5-VL模型..")
        
        # MPS设备初始化
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("⚠️ MPS不可用：当前PyTorch未启用MPS支持")
            else:
                print("⚠️ MPS不可用：需要macOS 12.3+或MPS设备")
        else:
            print("✅ MPS加速已启用")

        # 加载模型
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "/Users/xishi/Documents/comfy/ComfyUI/models/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map="mps",
            low_cpu_mem_usage=True,
            max_memory={0: "10GB"}
        )
        self.processor = AutoProcessor.from_pretrained(
            "/Users/xishi/Documents/comfy/ComfyUI/models/Qwen2.5-VL-7B-Instruct",
            min_pixels=256*28*28,
            max_pixels=1024*28*28,
            use_fast=True
        )
        print("✅ Qwen2.5-VL模型加载完成")

    def get_model(self):
        return self.model, self.processor

class Qwen2_5vVisionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "详细描述这张图片"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption_image"
    CATEGORY = "Qwen2.5v视觉模型"

    def __init__(self):
        self.model_manager = Qwen2_5vModelManager()
        self.model, self.processor = self.model_manager.get_model()
        
    def __del__(self):
        torch.mps.empty_cache()
        gc.collect()

    def caption_image(self, image, prompt):
        torch.mps.empty_cache()
        
        # 转换并调整图像尺寸
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # 检查并调整图像尺寸
        max_size = 1024
        if max(img.size) > max_size:
            print(f"⏳ 调整图像尺寸：{img.size} -> ", end="")
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"{img.size}")

        try:
            # 创建消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # 处理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("mps")

            # 生成输出
            generated_ids = self.model.generate(**inputs, max_new_tokens=1536)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return (output_text[0],)
        except Exception as e:
            print(f"❌ 推理失败: {str(e)}")
            return ("推理失败",)
        finally:
            torch.mps.empty_cache()
            del img
            gc.collect()

class Qwen2_5vTranslateNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "target_lang": (["en", "ja", "ko", "fr", "de"], {"default": "en"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("translated_text",)
    FUNCTION = "translate_text"
    CATEGORY = "Qwen2.5v视觉模型"

    def __init__(self):
        self.model_manager = Qwen2_5vModelManager()
        self.model, self.processor = self.model_manager.get_model()
        
    def __del__(self):
        torch.mps.empty_cache()
        gc.collect()

    def translate_text(self, text, target_lang):
        torch.mps.empty_cache()
        
        try:
            # 创建翻译提示
            prompt = f"将以下中文翻译成{target_lang}：{text}"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # 处理输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            ).to("mps")

            # 生成输出
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return (output_text[0],)
        except Exception as e:
            print(f"❌ 翻译失败: {str(e)}")
            return (text,)
        finally:
            torch.mps.empty_cache()
            gc.collect()

NODE_CLASS_MAPPINGS = {
    "Qwen2_5vVisionNode": Qwen2_5vVisionNode,
    "Qwen2_5vTranslateNode": Qwen2_5vTranslateNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2_5vVisionNode": "Qwen2.5v视觉模型",
    "Qwen2_5vTranslateNode": "Qwen2.5v翻译"
}