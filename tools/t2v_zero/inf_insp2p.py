# from huggingface_hub import hf_hub_download

# filename = "__assets__/pix2pix video/camel.mp4"
# repo_id = "PAIR/Text2Video-Zero"
# video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)


from PIL import Image
import imageio

reader = imageio.get_reader("/mnt/share_disk/leiyayun/data/hozon_neta/result/mp4s/4/ori_front_30.mp4", "ffmpeg")
frame_count = 30
video = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]



import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "/mnt/share_disk/LIV/generation_group/models/diffusers/public/instruct-pix2pix"
model_id = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/instructpix2pix/model/INS-HM-V0.4.2-5000"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))

prompt = "make it night"
result = pipe(prompt=[prompt] * len(video), image=video).images
imageio.mimsave("edited_video_inshm.mp4", result, fps=4)