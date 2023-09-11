import os 
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from diffusers import StableDiffusionControlNetInstructPix2PixPipeline
import PIL

image_path = "/mnt/ve_share/songyuhao/generation/data/test/v0.0/city14.png"

def preprocess_image(url):
    # image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    # image = image.resize((1024, 576))
    # image = image.resize((512, 512))
    
    return image

image = preprocess_image(image_path)


import cv2
from PIL import Image
import numpy as np
import torch
from diffusers import ControlNetModel

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)
canny_image.save("canny.png")

controlnet = ControlNetModel.from_pretrained("/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/control_v11p_sd15_canny",)
pipe = StableDiffusionControlNetInstructPix2PixPipeline.from_pretrained(
    "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/instructpix2pix/model/INS-HM-V0.4.0-5000", controlnet=controlnet,
).to("cuda")

# pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

out_image = pipe(
    "make it night", image=image, num_inference_steps=50, image_guidance_scale=1.5, guidance_scale=7, control_image=canny_image, controlnet_conditioning_scale=0.0
).images[0]

out_image.save("./hahaha_ori.png")