import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UniPCMultistepScheduler
import os
import cv2
from tqdm import tqdm
import numpy as np


def preprocess_image(url):
    # image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((512, 512))
    # image = image.resize((512, 512))
    
    return image

prompts = ["make it rainy"]
# model_names = ["INS-Base", "INS-HM-NIGHT-V0.0.0", "INS-HM-NIGHT-V0.0.1", "INS-HM-NIGHT-V0.1.0"]
model_names = ["INS-Base", "INS-HM-SNOWY-V0.0.0", "INS-HM-SNOWY-V0.0.1", "INS-HM-SNOWY-V0.1.0", "INS-HM-NIGHT-V0.0.0", "INS-HM-NIGHT-V0.0.1", "INS-HM-NIGHT-V0.1.0"]
model_dir = "/mnt/ve_share/generation/models/online/diffusions/res/instructpix2pix/prompt-to-prompt"
combine = True
test_path = '/mnt/ve_share/generation/data/train/diffusions/test_20'
image_paths = []
for foldername, subfolders, filenames in os.walk(test_path):
    for filename in filenames:
        # Get the full path of the file
        file_path = os.path.join(foldername, filename)
        image_paths.append(file_path)
        
n = len(image_paths)
print(n)

for ind, model_name in enumerate(model_names):
    print(model_name)
    res_dir = "/mnt/ve_share/generation/data/result/diffusions/vis/instructpix2pix/NORMAL_512/%s" % model_name
    os.makedirs(res_dir, exist_ok=True)
    
    if model_name == "INS-Base":
        model_id = "/mnt/ve_share/generation/models/online/diffusions/base/instruct-pix2pix"
    else:
        model_id= "%s/%s" % (model_dir, model_name)
        
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    for prompt in prompts:
        img_lst = []
        res_dir_p = "%s/%s" % (res_dir, "_".join(prompt.split(" ")))
        os.makedirs(res_dir_p, exist_ok=True)
        
        file_count = 0
        for _, _, files in os.walk(res_dir_p):
            file_count += len(files)
        
        if file_count >= n:        
            continue
        
        for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            test_image = preprocess_image(image_path)
            image = pipe(prompt, image=test_image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images[0]
            res_id = "%s/%d.png" % (res_dir_p, i)
            
            if combine:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
                test_image = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR) 
                im_combined = cv2.hconcat([test_image, image])
                cv2.imwrite(res_id, im_combined)

            else:
                image.save(res_id)
                