from diffusers import StableDiffusionPipeline
import torch

model_id = "/root/diffusers/textual_inversion_cat"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A <cat-toy> driving a car"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("/root/diffusers/vis/cat-backpack.png")