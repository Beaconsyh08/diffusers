import copy
import os
import sys
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import PIL
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, UniPCMultistepScheduler, UNet2DConditionModel
import os
from img_blur_processor import IMGBlurProcessor

sys.path.insert(0, "/root/GenLane/Mask2Former/demo")

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(str(Path(current_dir).parent))
sys.path.append(current_dir)


app = Flask(__name__)
CORS(app)

model_id = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/res/instructpix2pix/model/INS-HM-V0.3.0-5000"
model_id = "/share/songyuhao/generation/models/online/diffusions/res/instructpix2pix/model/INS-HM-V0.3.0-5000"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")


def preprocess_image(url):
    # image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.Image.open(url)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize((1024, 576))

    return image


@app.route('/backend/algorithm/global/playground', methods=['POST', 'GET'])
def playground():
    """
    input data example:
        {
        "header": {
            "timestamp": 1688636992401453
        },
        "message": {
            "scene": "rainy",
            "image": "xxxxxxxxxxxxxx.png",
        }
    """

    input_data = request.get_json()
    message = input_data['message']
    prompt = "make it " + message['scene']
    image_path = message['image'] + "/" if not message['image'].startswith("/") else image_path
    output_data = copy.deepcopy(input_data)

    if message['scene'] == "blur_lane":
        # img_blur_processor = IMGBlurProcessor(["/mnt/ve_share/songyuhao/generation/data/test/exp169/aaa.jpg"], 0.3, 5, "/mnt/ve_share/songyuhao/generation/data/genlan_test", "Gaussian", False)
        output_folder = "/root/result/GenLane"
        os.makedirs(output_folder, exist_ok=True)
        img_blur_processor = IMGBlurProcessor(image_path, 0.3, 5, output_folder, "Gaussian", True)
        image = img_blur_processor.run()
    else:
        test_image = preprocess_image(image_path)
        image = pipe(prompt, image=test_image, num_inference_steps=50, image_guidance_scale=1.5, guidance_scale=7).images[0]
        
    output_data['output'] = {"image": image}
    print(output_data)
    return jsonify(output_data)


@app.route('/backend/algorithm/global/factory', methods=['POST', 'GET'])
def factory():
    input_data = request.get_json()
    message = input_data['message']
    prompt = "make it " + message['scene']
    output_data = copy.deepcopy(input_data)

    pass
    return {}


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
