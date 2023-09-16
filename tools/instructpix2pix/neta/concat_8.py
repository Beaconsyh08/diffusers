

from PIL import Image, ImageDraw
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm


def concat_8(image_group, save_path):
    # Load your 8 images
    back = Image.open(image_group[0])
    back_left = Image.open(image_group[1])
    back_right = Image.open(image_group[2])
    front_left = Image.open(image_group[3])
    front_narrow = Image.open(image_group[4])
    front_right = Image.open(image_group[5])
    front_wide = Image.open(image_group[6])
    front_wide_crop = Image.open(image_group[7])

    # Create a blank image (you can specify the size)
    blank_image = Image.new('RGB', (back.width, back.height), color='black')

    images = [front_narrow, front_wide_crop, front_wide, front_left, blank_image, front_right, back_left, back, back_right]

    # Calculate the dimensions of the final image
    num_cols = 3
    num_rows = 3
    width = num_cols * back.width
    height = num_rows * back.height

    # Create a new blank image with the calculated dimensions
    final_image = Image.new('RGB', (width, height))

    # Paste the images into the final image to create the grid
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            final_image.paste(images[index], (j * back.width, i * back.height))

    # Save the final image
    final_image.save(save_path)


root_path = "/mnt/share_disk/leiyayun/data/hozon_neta/result/neta_f50_cn"
save_root = "/mnt/share_disk/leiyayun/data/hozon_neta/result/neta_f50_cn_combine"
os.makedirs(save_root, exist_ok=True)


prompts = os.listdir(root_path)

image_paths = {key: defaultdict(list) for key in prompts}

for basename, subname, filenames in os.walk(root_path):
    if len(basename.split("/")) == len(root_path.split("/")) + 2:
        bundle_id = basename.split("/")[-1]
        scene = basename.split("/")[-2]
        for filename in filenames:
            image_paths[scene][bundle_id] += ["%s/%s" % (basename, filename)]
            
# print(image_paths)

print(save_root)
for k, v in tqdm(image_paths.items()):
    scene = k
    save_root_2 ="%s/%s" % (save_root, scene)
    os.makedirs(save_root_2, exist_ok=True)
    for k2, v2 in tqdm(v.items()):
        bundle_id = k2
        images = v2
        save_path = "%s/%s.png" % (save_root_2, bundle_id)
        concat_8(images, save_path)


# # Display the final image
# final_image.show()

