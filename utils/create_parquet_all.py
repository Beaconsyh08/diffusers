import pandas as pd
import random
import os
from tqdm import tqdm

P2P_PATH = "/mnt/share_disk/syh/data/prompt_to_prompt/index.txt"
SCENE = "all"
PARA = "0.80_0.80_2.00"
SIZE = 8000
PARQUET_PATH = "/mnt/ve_share/generation/data/train/diffusions/parquet/%s_%s_%d" % (SCENE, PARA, SIZE*4)
os.makedirs(PARQUET_PATH, exist_ok=True)
PARQUET_PATH = "%s/p.parquet" % PARQUET_PATH


def prepare(input_image, edited_image, SIZE, scene):
    final_input_image, final_edited_image =[], []
    
    final_edited_image = sorted(edited_image, key=lambda x: x[-3])
    final_input_image = sorted(input_image, key=lambda x: x[-3])

    assert len(final_edited_image) == len(final_input_image), print(len(final_edited_image), len(final_input_image))

    sampled_items = random.sample(list(zip(final_input_image, final_edited_image)), SIZE)

    # Unpack the sampled items into separate lists
    final_input_image, final_edited_image = zip(*sampled_items)
    final_input_image = ["/".join(_) for _ in final_input_image]
    final_edited_image = ["/".join(_) for _ in final_edited_image]

    edit_prompt = ["make it %s" % scene for _ in range(SIZE)]
    assert len(final_edited_image) == len(final_input_image) == len(edit_prompt)
    
    return final_input_image, edit_prompt, final_edited_image


with open (P2P_PATH, "r") as input_file:
    paths = [_.strip().split("/") for _ in input_file.readlines()]
    
# -5: refine/replce
# -4: scene
# -3: id
# -2: parameter

edited_image_night, input_image_night, edited_image_snowy, input_image_snowy, edited_image_rainy, input_image_rainy, edited_image_foggy, input_image_foggy = [], [], [], [], [], [], [], []
for path in tqdm(paths):
    mode = path[-5]
    scene = path[-4]
    id = path[-3]
    parameter = path[-2]
    prompt = path[-1]
    if scene == "night" and mode == "replace_blend_reweight" and parameter == PARA:
        if "at_daytime" not in prompt and "at_nighttime" not in prompt:
            if scene[:-1] in prompt:
                edited_image_night.append(path)
            else:
                input_image_night.append(path)
    elif scene == "snowy" and mode == "refine_blend_reweight" and parameter == PARA:
        if scene[:-1] in prompt:
            edited_image_snowy.append(path)
        else:
            input_image_snowy.append(path)
    elif scene == "rainy" and parameter == PARA:
        if scene[:-1] in prompt:
            edited_image_rainy.append(path)
        else:
            input_image_rainy.append(path)
    elif scene == "foggy" and parameter == PARA:
        if scene[:-1] in prompt:
            edited_image_foggy.append(path)
        else:
            input_image_foggy.append(path)
            
n1, n2, n3 = prepare(input_image_night, edited_image_night, SIZE, "night")
s1, s2, s3 = prepare(input_image_snowy, edited_image_snowy, SIZE, "snowy")
r1, r2, r3 = prepare(input_image_rainy, edited_image_rainy, SIZE, "rainy")
f1, f2, f3 = prepare(input_image_foggy, edited_image_foggy, SIZE, "foggy")

input_image = n1 + s1 + r1 + f1
edit_prompt = n2 + s2 + r2 + f2
edited_image = n3 + s3 + r3 + f3

print(input_image[:10])
print(edit_prompt[:10])
print(edited_image[:10])

print(input_image[-10:])
print(edit_prompt[-10:])
print(edited_image[-10:])

print(len(input_image))


# Create a DataFrame with your data
data = {
    'input_image': input_image,
    'edit_prompt': edit_prompt,
    'edited_image': edited_image,
}

df = pd.DataFrame(data)

# Save the DataFrame as a Parquet file
df.to_parquet(PARQUET_PATH)
print(PARQUET_PATH)
