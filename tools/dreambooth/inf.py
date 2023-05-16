# from diffusers import DiffusionPipeline, UNet2DConditionModel
# from transformers import CLIPTextModel
# import torch

# # Load the pipeline with the same arguments (model, revision) that were used for training
# model_id = "CompVis/stable-diffusion-v1-4"

# unet = UNet2DConditionModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/unet")

# # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
# text_encoder = CLIPTextModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/text_encoder")

# pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
# pipeline.to("cuda")

# # Perform inference, or save, or push to the hub
# pipeline.save_pretrained("dreambooth-pipeline")


from diffusers import DiffusionPipeline
import torch
import os
import cv2
import numpy as np  

prompts = ["a street filled with lots of traffic at night time with lights on and cars driving down the street and a building in the background", 
           "a street that covered by heavy snow, filled with lots of traffic and cars driving down the street and a building in the background, 4k",
           "a street filled with lots of traffic and cars driving down the street and a building in the background, 4k",
           "a street filled with lots of traffic and cars driving down the street and a building in the background, rainy, 4k",
           "a photo of a dog"]

# prompts = ["a street filled with lots of traffic at night time with lights on and cars driving down the street and a building in the background, in the style of haomo", 
#            "a street that covered by heavy snow, filled with lots of traffic and cars driving down the street and a building in the background, in the style of haomo, 4k",
#            "a street filled with lots of traffic and cars driving down the street and a building in the background, in the style of haomo, 4k",
#            "a street filled with lots of traffic and cars driving down the street and a building in the background, rainy, in the style of haomo, 4k",
#            "a photo of a dog, in the style of haomo"]

model_names = ["SD-HM-V1.0"]

# model_dir = "./res/finetune/dreambooth" 
model_dir = "/mnt/ve_share/generation/models/online/diffusions/res/finetune/dreambooth"
n = 48
combine = True

for ind, model_name in enumerate(model_names):
    res_dir = "./vis/dreambooth/%s" % model_name
    os.makedirs(res_dir, exist_ok=True)
    model_id= "%s/%s" % (model_dir, model_name)
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    
    for prompt in prompts:
        img_lst = []
        res_dir_p = "%s/%s" % (res_dir, "_".join(prompt.split(" ")))
        os.makedirs(res_dir_p, exist_ok=True)
        
        for i in range(n):
            image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
            res_id = "%s/%d.png" % (res_dir_p, i)
            image.save(res_id)
                
            if combine:
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) 
                img_lst.append(image)

        if combine:
            list_of_lists = [img_lst[i : i + 4] for i in range(0, len(img_lst), 4)]
            im_combined = cv2.vconcat([cv2.hconcat(_) for _ in list_of_lists])
            res_id = "%s/combined.png" % (res_dir_p)
            cv2.imwrite(res_id, im_combined)
            print(res_id)