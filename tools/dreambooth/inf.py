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

prompts = ["a photo of sks night traffic scene", ]
# prompts = ["a photo of a dog", "a photo of a dog", ]
model_names = ["haomo_night_sks_200x",]
model_dir = "./res/finetune/dreambooth" 
n = 6

for ind, model_name in enumerate(model_names):
    res_dir = "./vis/dreambooth/%s" % model_name
    os.makedirs(res_dir, exist_ok=True)
    model_id= "%s/%s" % (model_dir, model_name)
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    prompt = prompts[ind]
    
    for i in range(n):
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        res_id = "%s/%s_%d.png" % (res_dir, "_".join(prompt.split(" ")), i)
        image.save(res_id)
        print(res_id)
