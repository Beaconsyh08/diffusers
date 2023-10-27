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
    image = image.resize((1024, 576))
    # image = image.resize((1920, 1080))
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

CONTROL_SCALE = 0.3
combine = False
draw_text = False
torch_dtype = torch.float32
text_dict = {"dawn": "清晨", "dusk": "黄昏", "night": "夜晚", "rainy": "雨天", "snowy": "雪天", "cloudy": "多云", "foggy": "雾天", "contre-jour": "逆光"}
# prompts = ["make it night", ("make it night", "daytime"), ("make it night", "sunshine"), "make it contre-jour", "make it rainy", "make it rainy night", "make it night rainy", "make it backlight"]
# prompts = ["make it rainy",  "make it contre-jour"]
prompts = ["make it night"]

# model_names = ["INS-HM-V0.4.0-5000", "INS-HM-V0.3.0-5000", "INS-HM-V0.4.3-5000"]
# model_names = ["INS-HM-V0.3.0-5000", "INS-HM-V0.4.0-5000", "INS-HM-V0.4.3-5000"]
model_names = ["INS-HM-V0.4.2-5000"]

model_dir = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/instructpix2pix/model"

# test_path = "/mnt/share_disk/leiyayun/data/hozon_neta/EP40-PVS-42_20230420_141710/1681897682_030556160-EP40-PVS-42_EP40_MDC_0930_1215_2023-04-19-09-47-52_25"
# test_path = "/mnt/share_disk/leiyayun/data/hozon_neta/EP40-PVS-42_20230420_141710/1681899865_963508992-EP40-PVS-42_EP40_MDC_0930_1215_2023-04-19-10-24-22_98"
# test_path = "/mnt/share_disk/leiyayun/data/hozon_neta/EP40-PVS-42_20230420_141710/1681900271_051031040-EP40-PVS-42_EP40_MDC_0930_1215_2023-04-19-10-30-53_111"
test_path = "/mnt/share_disk/leiyayun/data/hozon_neta/EP40-PVS-42_20230420_141710"

# test_path = "/mnt/share_disk/leiyayun/data/hozon_neta/result/test_1_nc/INS-HM-V0.4.3-5000/rainy"
res_root = "/mnt/share_disk/leiyayun/data/hozon_neta/result/neta_cn"

image_paths, folders = [], []
for foldername, subfolders, filenames in os.walk(test_path):
    for filename in filenames:
        # Get the full path of the file
        folders.append(foldername.split("/")[-1])
        file_path = os.path.join(foldername, filename)
        if file_path.endswith(".jpg") or file_path.endswith(".png"):
            image_paths.append(file_path)
        
n = len(image_paths)
image_paths.sort()
image_paths = image_paths[4000:]
print(n)

for ind, model_name in enumerate(model_names):
    print(model_name)
    # res_dir = "%s/%s" % (res_root,)

    
    if model_name == "INS-Base":
        model_id = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/instruct-pix2pix"
    else:
        model_id= "%s/%s" % (model_dir, model_name)
    
    controlnet = ControlNetModel.from_pretrained("/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/control_v11p_sd15_canny", torch_dtype=torch_dtype).to("cuda")
    if "/" in model_name:
        unet = UNet2DConditionModel.from_pretrained("%s/unet_ema" % model_id)
        model_id_true = "/".join(model_id.split("/")[:-1])
        pipe = StableDiffusionControlNetInstructPix2PixPipeline.from_pretrained(model_id_true, unet=unet, controlnet=controlnet, torch_dtype=torch_dtype).to("cuda")
        
    else:
        pipe = StableDiffusionControlNetInstructPix2PixPipeline.from_pretrained(model_id, controlnet=controlnet).to("cuda")
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    pipe.safety_checker = lambda images, **kwargs: (images, [False] * len(images))
    
    for prompt in prompts:
        print(prompt)
        img_lst = []
        prompt_neg = None
        if type(prompt) == tuple:
            prompt_neg = prompt[1]
            prompt = prompt[0]
            print("neg", prompt_neg)
            
            res_dir_p = "%s/%s_neg_%s" % (res_root, "_".join(prompt.split(" ")), "_".join(prompt_neg.split(" ")))
        else:
            res_dir_p = "%s/%s_%.2f-0.4.2" % (res_root, "_".join(prompt.split(" ")), CONTROL_SCALE)
        os.makedirs(res_dir_p, exist_ok=True)
        print(res_dir_p)
        
        file_count = 0
        for _, _, files in os.walk(res_dir_p):
            file_count += len(files)
        
        # if file_count >= n:        
        #     continue
        
        for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
            if image_path[-5] == "e":
                continue    
            test_image = preprocess_image(image_path)
            control_image = preprocess_canny_image(test_image)
            image = pipe(prompt, image=test_image, num_inference_steps=50, image_guidance_scale=1.5, guidance_scale=7, negative_prompt=prompt_neg, control_image=control_image, controlnet_conditioning_scale=CONTROL_SCALE).images[0]
            
            res_dir = "%s/%s" % (res_dir_p, image_path.split(".")[0].split("/")[-2])
            os.makedirs(res_dir, exist_ok=True)
            
            res_id = "%s/%s" % (res_dir, image_path.split("/")[-1])
    
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
                