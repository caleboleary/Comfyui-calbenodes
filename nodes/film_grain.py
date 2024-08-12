import numpy as np
import torch

class FilmGrain:
    def __init__(self):
        self.type = "FilmGrain"
        self.name = "Add Film Grain"
        self.description = "Adds film grain effect to an image"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 0.07,
                    "min": 0.01,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_film_grain"
    CATEGORY = "calbenodes"

    def add_film_grain(self, image, intensity):
        # Ensure we're working with a batch of images
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Move to CPU for numpy operations
        image_np = image.cpu().numpy()
        
        # Generate noise for each image in the batch
        noised_images = []
        for img in image_np:
            # Convert to range [0, 255]
            img = (img * 255).astype(np.float32)
            
            # Generate unique noise for each pixel
            h, w, c = img.shape
            noise = np.random.normal(0, 50, (h, w, c)).astype(np.float32)
            
            # Normalize noise to [-1, 1]
            noise = 2 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise)) - 1
            
            # Apply noise
            noised_img = (1 - intensity) * img + intensity * noise * 255
            noised_img = np.clip(noised_img, 0, 255).astype(np.float32) / 255.0
            
            noised_images.append(noised_img)
        
        # Stack the processed images back into a batch
        result = np.stack(noised_images)
        
        # Convert back to torch tensor
        result_tensor = torch.from_numpy(result).to(image.device)
        
        return (result_tensor,)
