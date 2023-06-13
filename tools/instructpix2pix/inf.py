import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline

model_id = "/mnt/ve_share/generation/models/online/diffusions/base/instruct-pix2pix"
model_id = "/mnt/ve_share/generation/models/online/diffusions/res/instruct-pix2pix-test"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
url = "/root/diffusers/data/i2i/source/mountain.png"


def preprocess_image(url):
    # image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


image = preprocess_image(url)

prompt = "make it midnight"
images = pipe(prompt, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
images[0].save("/root/diffusers/data/i2i/inf/midnight_mountains_2.png")