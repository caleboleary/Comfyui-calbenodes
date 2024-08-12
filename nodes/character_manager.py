import os
import random
import json
import math
from PIL import Image
import numpy as np
import torch
from nodes import LoraLoader
import folder_paths

class CharacterManagerNode:
    def __init__(self):
        self.characters = self.load_characters()

    @classmethod
    def INPUT_TYPES(cls):
        characters = ["Random", "New Character"] + list(cls().characters.keys())
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "character": (characters,),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "new_name": ("STRING", {"default": ""}),
                "lora_path": ("STRING", {"default": ""}),
                "face_images_dir": ("STRING", {"default": ""}),
                "preferred_face_image": ("STRING", {"default": ""}),
                "activation_text": ("STRING", {"default": ""}),
                "description": ("STRING", {"default": ""}),
                "negative_prompt": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "STRING", "IMAGE", "IMAGE", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("model", "clip", "lora_activation", "description", "negative_prompt", "preferred_face", "random_face", "face_grid", "character_name", "seed")
    FUNCTION = "process_character"
    CATEGORY = "calbenodes"

    def load_characters(self):
        characters_path = os.path.join(folder_paths.base_path, "characters.json")
        if os.path.exists(characters_path):
            with open(characters_path, "r") as f:
                return json.load(f)
        return {}

    def save_characters(self):
        characters_path = os.path.join(folder_paths.base_path, "characters.json")
        with open(characters_path, "w") as f:
            json.dump(self.characters, f, indent=2)

    def process_character(self, model, clip, character, lora_strength, seed, new_name="", lora_path="", face_images_dir="", 
                          preferred_face_image="", activation_text="", description="", negative_prompt=""):
        # Set the seed for this execution
        random.seed(seed)

        print(f"character: {character}")

        if character == "Random":
            character = random.choice(list(self.characters.keys()))
            print(f"Randomly selected character: {character}")

        if character == "New Character":
            if not new_name:
                raise ValueError("New name is required when creating a new character.")
            self.characters[new_name] = {
                "lora_path": lora_path,
                "face_images_dir": face_images_dir,
                "preferred_face_image": preferred_face_image,
                "activation_text": activation_text,
                "description": description,
                "negative_prompt": negative_prompt
            }
            self.save_characters()
            character = new_name
            print(f"Created new character: {character}")
        
        char_data = self.characters[character]

        # Apply LoRA
        if char_data["lora_path"] and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, char_data["lora_path"], lora_strength, lora_strength)

        # Get preferred face image
        preferred_face = self.get_preferred_face_image(char_data.get("preferred_face_image", ""), char_data["face_images_dir"])

        # Get random face image
        random_face = self.get_random_face(char_data["face_images_dir"])

        # Generate face grid
        face_grid = self.generate_grid(char_data["face_images_dir"])

        return (
            model,
            clip,
            char_data["activation_text"],
            char_data["description"],
            char_data.get("negative_prompt", ""),
            preferred_face,
            random_face,
            face_grid,
            character,
            seed 
        )

    def get_preferred_face_image(self, preferred_face_image, face_images_dir=""):
        if preferred_face_image and os.path.exists(preferred_face_image):
            image = Image.open(preferred_face_image).convert('RGB')
            image = self.resize_image(image, 256)  # Resize to a standard size
            return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
        return self.get_random_face(face_images_dir)

    def get_random_face(self, face_images_dir):
        if not face_images_dir:
            return torch.zeros((1, 3, 256, 256))
        image_files = [f for f in os.listdir(face_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        if not image_files:
            return torch.zeros((1, 3, 256, 256))
        
        random_image = Image.open(os.path.join(face_images_dir, random.choice(image_files))).convert('RGB')
        random_image = self.resize_image(random_image, 256)  # Resize to a standard size
        return torch.from_numpy(np.array(random_image).astype(np.float32) / 255.0).unsqueeze(0)


    def generate_grid(self, directory, max_images=256, max_size=768):
        if not directory:
            return torch.zeros((1, 3, max_size, max_size))
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        if not image_files:
            return torch.zeros((1, 3, max_size, max_size))
        
        image_files = image_files[:max_images]
        
        images = [self.resize_image(Image.open(os.path.join(directory, img)).convert('RGB'), max_size) for img in image_files]
        
        n = len(images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        grid_width = cols * max_size
        grid_height = rows * max_size
        grid_img = Image.new('RGB', (grid_width, grid_height))
        
        for i, img in enumerate(images):
            x = (i % cols) * max_size
            y = (i // cols) * max_size
            grid_img.paste(img, (x, y))
        
        grid_np = np.array(grid_img).astype(np.float32) / 255.0
        return torch.from_numpy(grid_np)[None,]

    def resize_image(self, img, max_size):
        width, height = img.size
        if width > height:
            if width > max_size:
                height = int(max_size * height / width)
                width = max_size
        else:
            if height > max_size:
                width = int(max_size * width / height)
                height = max_size
        return img.resize((width, height), Image.LANCZOS)