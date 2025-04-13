from .samone.noise import Latent
from .samone.sampler import SamplerOne

NODE_CLASS_MAPPINGS = {
    "Latent": Latent,
    "SamplerOne": SamplerOne,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
