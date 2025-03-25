import torch
import numpy as np
from PIL import Image, ImageOps
import nodes
import folder_paths

class ImageResizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "width": ("INT", {"default": 1280, "min": 1, "max": 10000}),
                "height": ("INT", {"default": 1600, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "resize_image"
    CATEGORY = "image/processing"

    def resize_image(self, image, width=1280, height=1600):
        # Convert tensor to PIL Image
        i = 255. * image[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        # Calculate aspect ratio
        orig_width, orig_height = img.size
        ratio = min(width/orig_width, height/orig_height)
        
        # Resize while preserving aspect ratio and details
        new_size = (int(orig_width * ratio), int(orig_height * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        
        # Create new image with target size and paste the resized image
        result = Image.new("RGB", (width, height))
        result.paste(img, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
        
        # Convert back to tensor
        result = np.array(result).astype(np.float32) / 255.0
        result = torch.from_numpy(result)[None,]
        
        return (result,)

NODE_CLASS_MAPPINGS = {
    "ImageResize": ImageResizeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResize": "Image Resize Node"
}