from diffusers import StableDiffusionPipeline
import torch

model_id = "./res/finetune/textual_inversion/cat_toy"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A <cat-toy> backpack"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("./vis/cat-backpack.png")