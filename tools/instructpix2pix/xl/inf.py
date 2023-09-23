import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from diffusers.utils import load_image

resolution = 768
image = load_image("/root/diffusers/Eiffel.jpg").resize((resolution, resolution)).convert('RGB')
edit_instruction = "make it night"

pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
    "/mnt/share_disk/LIV/generation_group/models/diffusers/public/sdxl-instructpix2pix-768", torch_dtype=torch.float16
).to("cuda")

edited_image = pipe(
    prompt=edit_instruction,
    image=image,
    height=resolution,
    width=resolution,
    guidance_scale=7.5,
    image_guidance_scale=1.5,
    num_inference_steps=50,
).images[0]
edited_image.save("./hahaxl.png")