import os 
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import PIL
import torch
from diffusers import StableDiffusionControlNetInstructPix2PixPipeline, UniPCMultistepScheduler, UNet2DConditionModel, ControlNetModel
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


def preprocess_image(url):
    # image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    # image = image.resize((1024, 576))
    # image = image.resize((512, 512))
    
    return image


def preprocess_canny_image(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    return canny_image

# prompts = ["make it dawn", 
#            "make it dusk", 
#            "make it night", 
#            "make it rainy", 
#            "make it snowy", 
#            "make it cloudy", 
#            "make it foggy", 
#            "make it contre-jour",
#            ("make it night", "daytime"),
#            ("make it night", "sunshine")]


# prompts = ["make it night", 
#            "make it rainy", 
#            "make it snowy", 
#            ("make it night", "daytime"),
#            ("make it night", "sunshine")]

prompts = ["make it happy"]

draw_text = False
text_dict = {"dawn": "清晨", "dusk": "黄昏", "night": "夜晚", "rainy": "雨天", "snowy": "雪天", "cloudy": "多云", "foggy": "雾天", "contre-jour": "逆光"}

# prompts = ["make it heavy snowy"]


# model_names = ["INS-HM-V0.4.3/checkpoint-5000", "INS-HM-V0.4.3/checkpoint-10000", "INS-HM-V0.4.4/checkpoint-5000", "INS-HM-V0.4.4/checkpoint-10000", "INS-HM-V0.4.4", "INS-HM-V0.4.3"]
# model_names = ["INS-HM-V0.4.0-5000", "INS-HM-V0.3.0-5000", "INS-HM-V0.4.3-5000"]
model_names = ["INS-HM-V0.4.3-5000"]

model_dir = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/instructpix2pix/model"
# model_dir = "/mnt/share_disk/songyuhao/models/online/diffusions/res/instructpix2pix/model"

combine = True

# test_path = '/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/test_lyy/INS-HM-V0.3.0-5000/rainy'
# test_path = "/mnt/ve_share/songyuhao/generation/data/test/kl"
test_path = "/mnt/ve_share/songyuhao/generation/data/test/v0.0"
res_root = "/mnt/ve_share/songyuhao/generation/data/result/diffusions/vis/instructpix2pix/cn_test"

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
    res_dir = "%s/%s" % (res_root, model_name.split("/")[0] + "-" + model_name.split("/")[-1].split("-")[-1]) if "/" in model_name else "%s/%s" % (res_root, model_name)
    os.makedirs(res_dir, exist_ok=True)
    
    if model_name == "INS-Base":
        model_id = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/instruct-pix2pix"
    else:
        model_id= "%s/%s" % (model_dir, model_name)
    
    controlnet = ControlNetModel.from_pretrained("/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/control_v11p_sd15_canny")
    if "/" in model_name:
        unet = UNet2DConditionModel.from_pretrained("%s/unet_ema" % model_id)
        model_id_true = "/".join(model_id.split("/")[:-1])
        pipe = StableDiffusionControlNetInstructPix2PixPipeline.from_pretrained(model_id_true, unet=unet, controlnet=controlnet).to("cuda")
        
    else:
        pipe = StableDiffusionControlNetInstructPix2PixPipeline.from_pretrained(model_id,  controlnet=controlnet).to("cuda")
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    
    for prompt in prompts:
        print(prompt)
        img_lst = []
        prompt_neg = None
        if type(prompt) == tuple:
            prompt_neg = prompt[1]
            prompt = prompt[0]
            res_dir_p = "%s/%s_neg_%s" % (res_dir, "_".join(prompt.split(" ")), "_".join(prompt_neg.split(" ")))
        else:
            res_dir_p = "%s/%s" % (res_dir, "_".join(prompt.split(" ")))
        os.makedirs(res_dir_p, exist_ok=True)
        
        file_count = 0
        for _, _, files in os.walk(res_dir_p):
            file_count += len(files)
        
        if file_count >= n:        
            continue
        
        for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            test_image = preprocess_image(image_path)
            control_image = preprocess_canny_image(test_image)
            image = pipe(prompt, image=test_image, num_inference_steps=50, image_guidance_scale=1.5, guidance_scale=7, negative_prompt=prompt_neg, control_image=control_image, controlnet_conditioning_scale=0.5).images[0]
            res_id = "%s/%d.png" % (res_dir_p, i)
            
            if draw_text:
                scene = prompt.split(" ")[-1]
                I1 = ImageDraw.Draw(image)
                myFont = ImageFont.truetype('/mnt/ve_share/songyuhao/fonts/simsun.ttc', 65, encoding="utf-8")
                I1.text((10, 10), text_dict[scene], font=myFont, fill =(255, 0, 0))
                
            if combine:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
                test_image = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR) 
                im_combined = cv2.hconcat([test_image, image])
                cv2.imwrite(res_id, im_combined)

            else:
                image.save(res_id)
                