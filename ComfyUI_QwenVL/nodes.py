import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import folder_paths
import subprocess
import uuid


def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze()
    return img


class Qwen2VL:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.bf16_support = torch.cuda.get_device_capability(self.device)[0] >= 8
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.bf16_support = True
        else:
            self.device = torch.device("cpu")
            self.bf16_support = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-VL-3B-Instruct",
                        "Qwen2.5-VL-7B-Instruct",
                    ],
                    {"default": "Qwen2.5-VL-3B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        text,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
        image=None,
        video_path=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)
        model_id = f"qwen/{model}"
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.processor is None:
            # Define min_pixels and max_pixels:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            min_pixels = 256*28*28
            max_pixels = 1024*28*28 

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="mps" if self.device.type == "mps" else "auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]

            if video_path:
                print("deal video_path", video_path)
                # 使用FFmpeg处理视频
                unique_id = uuid.uuid4().hex  # 生成唯一标识符
                processed_video_path = f"/tmp/processed_video_{unique_id}.mp4"  # 临时文件路径
                ffmpeg_command = [
                    "ffmpeg",
                    "-i", video_path,
                    "-vf", "fps=1,scale='min(256,iw)':min'(256,ih)':force_original_aspect_ratio=decrease",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",
                    processed_video_path
                ]
                subprocess.run(ffmpeg_command, check=True)

                # 添加处理后的视频信息到消息
                messages[0]["content"].insert(0, {
                    "type": "video",
                    "video": processed_video_path,
                })

            # 处理图像输入
            else:
                print("deal image")
                pil_image = tensor_to_pil(image)
                # 确保图像尺寸适合MPS
                if self.device.type == "mps":
                    max_pixels = 512 * 512  # MPS设备上更严格的像素限制
                    current_pixels = pil_image.width * pil_image.height
                    
                    # 渐进式缩放直到满足要求
                    while current_pixels > max_pixels:
                        scale_factor = (max_pixels / current_pixels) ** 0.5
                        new_width = max(64, int(pil_image.width * scale_factor))
                        new_height = max(64, int(pil_image.height * scale_factor))
                        pil_image = pil_image.resize(
                            (new_width, new_height),
                            Image.LANCZOS
                        )
                        current_pixels = new_width * new_height
                        print(f"Scaled image to {new_width}x{new_height} for MPS compatibility")
                messages[0]["content"].insert(0, {
                    "type": "image",
                    "image": pil_image,
                })

            # 准备输入
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                print("deal messages", messages)
                image_inputs, video_inputs = process_vision_info(messages)
                
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                
                # 对MPS设备进行额外检查
                if self.device.type == "mps":
                    total_size = 0
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            tensor_size = v.numel() * v.element_size()
                            total_size += tensor_size
                            if tensor_size > 2**30:  # 单个张量超过1GB
                                return ("Error: Input tensor too large for MPS device (max 1GB per tensor)",)
                    if total_size > 2**30:  # 总输入超过1GB
                        return ("Error: Total input size too large for MPS device (max 1GB total)",)
            except Exception as e:
                return (f"Error during input preparation: {str(e)}",)
            # 确保张量在正确设备上
            inputs = inputs.to(self.device)

            # 推理 - 对MPS设备使用更保守的参数
            try:
                generate_kwargs = {
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                }
                if self.device.type == "mps":
                    generate_kwargs.update({
                        'max_length': min(max_new_tokens + inputs.input_ids.shape[-1], 2048),
                        'early_stopping': True,
                    })
                generated_ids = self.model.generate(**inputs, **generate_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    temperature=temperature,
                )
            except Exception as e:
                return (f"Error during model inference: {str(e)}",)

            if not keep_model_loaded:
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            # 删除临时视频文件
            if video_path:
                os.remove(processed_video_path)

            return result


class Qwen2:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.bf16_support = torch.cuda.get_device_capability(self.device)[0] >= 8
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.bf16_support = True
        else:
            self.device = torch.device("cpu")
            self.bf16_support = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-3B-Instruct",
                        "Qwen2.5-7B-Instruct",
                        "Qwen2.5-14B-Instruct",
                        "Qwen2.5-32B-Instruct",
                    ],
                    {"default": "Qwen2.5-7B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),  # add quantization type selection
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        system,
        prompt,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
    ):
        if not prompt.strip():
            return ("Error: Prompt input is empty.",)

        if seed != -1:
            torch.manual_seed(seed)
        model_id = f"qwen/{model}"
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
                local_dir_use_symlinks=False,
            )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="mps" if self.device.type == "mps" else "auto",
                quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.tokenizer.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )

            if not keep_model_loaded:
                del self.tokenizer
                del self.model
                self.tokenizer = None
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            return result