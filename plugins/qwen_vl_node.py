import comfy
import folder_paths
import torch
import requests
import json
from PIL import Image
import io
import numpy as np
import os

class QwenVLImageCaption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "caption_image"
    CATEGORY = "QwenVL"

    def caption_image(self, image, prompt):
        # Convert tensor to PIL Image
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Save image to temp file
        temp_dir = folder_paths.get_temp_directory()
        temp_file = os.path.join(temp_dir, "qwen_vl_temp.png")
        img.save(temp_file)
        
        # Prepare API request
        api_url = "http://localhost:8000/generate"
        payload = {
            "image_path": temp_file,
            "prompt": prompt
        }
        
        try:
            response = requests.post(api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            return (result["result"],)
        except Exception as e:
            print(f"Error calling QwenVL API: {str(e)}")
            return ("API call failed",)

NODE_CLASS_MAPPINGS = {
    "QwenVLImageCaption": QwenVLImageCaption
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenVLImageCaption": "QwenVL Image Caption"
}