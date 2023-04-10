import torch
from diffusers import StableDiffusionPipeline

# model_base = "runwayml/stable-diffusion-v1-5"
model_base = "/mnt/share_disk/lei/git/diffusers/local_models/stable-diffusion-v1-5"
model_path = "./res/finetune/dreambooth_lora/dog"

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
# image = pipe("A picture of a sks dog swimming", num_inference_steps=25, guidance_scale=7.5).images[0]
image.save("./vis/bucket-dog.png")