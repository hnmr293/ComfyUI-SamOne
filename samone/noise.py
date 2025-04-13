import torch

import comfy.model_management
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise


# noise 生成
class Latent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": ("STRING", {"tooltip": "Example:\n(1, 2, 3)\n1, 2, 3, 4"}),
                # "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "noise": ("NOISE",),
                "dtype": ("STRING", {"default": "float32", "values": ["float32", "float16", "bfloat16"]}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("noise", "latent_iamge")

    FUNCTION = "generate_noise"

    CATEGORY = "hnmr/samone"

    def _parse_shape(self, shape: str):
        shape = shape.strip("()").split(",")
        return tuple(map(int, shape))

    def generate_noise(self, shape: str, noise: Noise_EmptyNoise, dtype: str):
        try:
            s = self._parse_shape(shape)
        except ValueError:
            raise ValueError(f"Invalid shape format: {shape}. Expected format: (1, 2, 3) or 1, 2, 3, 4")

        if len(s) == 0:
            raise ValueError("Shape cannot be empty.")

        device = comfy.model_management.intermediate_device()
        dtype = getattr(torch, dtype)

        latent_image = {"samples": torch.zeros(s, device=device, dtype=dtype)}
        latent = {"samples": noise.generate_noise(latent_image)}
        return (latent, latent_image)
