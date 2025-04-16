import torch

from comfy.samplers import CFGGuider, Sampler
from comfy.model_base import BaseModel
import comfy.model_management


class SamplerOne:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigma_from": ("FLOAT", {"step": 0.0001}),
                "sigma_to": ("FLOAT", {"step": 0.0001}),
                "model_scaling_in": ("BOOLEAN", {"default": False, "tooltip": "do enable for input scaling"}),
            },
            "optional": {
                "latent_image": ("LATENT",),
                "sampler_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "LATENT", "LATENT")
    RETURN_NAMES = ("scaled_sample", "non_scaled_sample", "model_input", "model_output")

    FUNCTION = "sample"
    CATEGORY = "hnmr/samone"

    def sample(
        self,
        latent: dict,
        guider: CFGGuider,
        sampler: Sampler,
        sigma_from: float,
        sigma_to: float,
        model_scaling_in: bool,
        latent_image: dict | None = None,
        sampler_seed: int | None = None,
    ):
        noise: torch.Tensor = latent["samples"]
        latent_mask = latent["mask"] if "mask" in latent else None
        if latent_image is None:
            ref_latent = torch.zeros_like(noise)
        else:
            ref_latent = latent_image["samples"]

        if noise.shape != ref_latent.shape:
            raise ValueError("latent and latent_image must have the same shape")

        if noise.dtype != ref_latent.dtype:
            raise ValueError("latent and latent_image must have the same dtype")

        sigmas = torch.tensor([sigma_from, sigma_to], device=noise.device, dtype=torch.float32)

        # モデルへの入力をフックしてスケールしていない latent を取得する

        inner_model: BaseModel = guider.model_patcher.model
        model_sampling = inner_model.model_sampling
        orig_latent_in = inner_model.process_latent_in
        orig_latent_out = inner_model.process_latent_out
        orig_noise_scaling = model_sampling.noise_scaling
        orig_noise_scaling_inv = model_sampling.inverse_noise_scaling

        if not model_scaling_in:

            def dummy_latent_in(latent):
                return latent

            def dummy_noise_scaling(sigma, noise, latent_image, max_denoise=False):
                return noise

            inner_model.process_latent_in = dummy_latent_in
            model_sampling.noise_scaling = dummy_noise_scaling

        non_scaled_latent1 = None
        non_scaled_latent2 = None

        def dummy_latent_out(latent):
            nonlocal non_scaled_latent2
            assert non_scaled_latent2 is None, "non_scaled_latent should be None"
            non_scaled_latent2 = latent
            scaled_latent = orig_latent_out(latent)
            return scaled_latent

        def dummy_noise_scaling_inv(sigma, latent):
            nonlocal non_scaled_latent1
            assert non_scaled_latent1 is None, "non_scaled_latent should be None"
            non_scaled_latent1 = latent
            scaled_latent = orig_noise_scaling_inv(sigma, latent)
            return scaled_latent

        inner_model.process_latent_out = dummy_latent_out
        model_sampling.inverse_noise_scaling = dummy_noise_scaling_inv

        # モデルへの入力をフックして、モデルへの入力とモデルからの出力を取得する
        model_input = None
        model_output = None
        diffusion_model = inner_model.diffusion_model

        def hook_forward(module, input, output):
            # input := (xc, t, ...)
            nonlocal model_input, model_output
            model_input = input[0].clone().detach()
            model_output = output.clone().detach()

        hook_handle = diffusion_model.register_forward_hook(hook_forward)

        def dummy_callback(*args, **kwargs):
            pass

        try:
            samples = guider.sample(
                noise=noise,
                latent_image=ref_latent,
                sampler=sampler,
                sigmas=sigmas,
                denoise_mask=latent_mask,
                callback=dummy_callback,
                disable_pbar=True,
                seed=sampler_seed,
            )
        finally:
            if not model_scaling_in:
                inner_model.process_latent_in = orig_latent_in
                model_sampling.noise_scaling = orig_noise_scaling
            inner_model.process_latent_out = orig_latent_out
            model_sampling.inverse_noise_scaling = orig_noise_scaling_inv
            hook_handle.remove()

        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples

        non_scaled_out = latent.copy()
        non_scaled_out["samples"] = non_scaled_latent2

        model_input_ = {"samples": model_input}
        model_output_ = {"samples": model_output}

        return (out, non_scaled_out, model_input_, model_output_)
