# ComfyUI-SamOne - one-step sampling

[日本語 ver.](./README_ja.md)

A node for ComfyUI that advances sampling by just one step.

Nodes are added under `hnmr/samone`.

## Nodes

### SamplerOne

![node image](./assets/image.png)

A node that advances sampling by just one step.

| in/out | name | type | Description |
| --- | --- |--- | --- |
| in | latent | LATENT | Input a noisy latent |
| in | guider | GUIDER | This is something like CFGGuider |
| in | sampler | SAMPLER | Input the output of KSamplerSelect |
| in | sigma_from | FLOAT | Input the σ of the step (note this is not time) |
| in | sigma_to | FLOAT | Input the σ of the next step (note this is not time) |
| in | model_scaling_in | BOOLEAN | Set to True for initial steps |
| in [opt] | latent_image | LATENT | Used for img2img |
| in [opt] | sampler_seed | INT | Used when the sampler takes a seed |
| out | scaled_sample | LATENT | Output scaled for passing through VAE |
| out | non_scaled_sample | LATENT | Unscaled output for use in the next sampling |

### Latent

A convenient node that generates noisy latent by providing `shape` as a string.

## Example workflow

[sample workflow (JSON)](./assets/SamOneTest.json)

![sample image](./assets/sample.png)