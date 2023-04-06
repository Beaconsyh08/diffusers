import torch
from diffusers import StableDiffusionPipeline

model_base = "runwayml/stable-diffusion-v1-5"
model_path = "/root/diffusers/res/finetune/dreambooth/dog"

pipe = StableDiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16)

pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

image = pipe(
    "A picture of a sks dog in a bucket.",
    num_inference_steps=25,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 0.5},
).images[0]

image = pipe("A picture of a sks dog in a bucket.", num_inference_steps=25, guidance_scale=7.5).images[0]
image.save("bucket-dog.png")