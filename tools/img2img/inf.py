# !pip install opencv-python transformers accelerate
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch

import cv2
from PIL import Image

# download an image
image = load_image(
    "/mnt/ve_share/songyuhao/generation/data/train/diffusions/ins_20k/imgs/00000-0-10086323494849853.png"
).resize((512, 512))
np_image = np.array(image)

# get canny image
np_image = cv2.Canny(np_image, 100, 200)
np_image = np_image[:, :, None]
np_image = np.concatenate([np_image, np_image, np_image], axis=2)
canny_image = Image.fromarray(np_image)

# load control net and stable diffusion v1-5
controlnet = ControlNetModel.from_pretrained("/mnt/share_disk/LIV/generation_group/models/diffusers/public/control_v11p_sd15_canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/finetune/dreambooth/SD-HM-V0.4.0", controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# generate image
generator = torch.manual_seed(0)
image = pipe(
    "rainy, a car driving down a highway next to a forest filled with trees and bushes on a cloudy day with a blue sky",
    num_inference_steps=20,
    generator=generator,
    image=image,
    control_image=canny_image,
).images[0]
image.save("hahaha.png")