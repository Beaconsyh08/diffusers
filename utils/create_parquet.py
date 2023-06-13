import pandas as pd
import random
import os
from tqdm import tqdm

P2P_PATH = "/mnt/share_disk/syh/data/prompt_to_prompt/index.txt"
MODE = "replace_blend_reweight"
SCENE = "rainy"
PARA = "0.80_0.80_2.00"
SIZE = 1000
STREET = False
PARQUET_PATH = "/mnt/ve_share/generation/data/train/diffusions/parquet/%s_%s_%s_%d_street" % (MODE, SCENE, PARA, SIZE) if STREET else "/mnt/ve_share/generation/data/train/diffusions/parquet/%s_%s_%s_%d" % (MODE, SCENE, PARA, SIZE)
os.makedirs(PARQUET_PATH, exist_ok=True)
PARQUET_PATH = "%s/p.parquet" % PARQUET_PATH

with open (P2P_PATH, "r") as input_file:
    paths = [_.strip().split("/") for _ in input_file.readlines()]
    
# -5: refine/replce
# -4: scene
# -3: id
# -2: parameter

input_image, edited_image = [], []
for path in tqdm(paths):
    mode = path[-5]
    scene = path[-4]
    id = path[-3]
    parameter = path[-2]
    prompt = path[-1]
    if mode == MODE and scene == SCENE and parameter == PARA:
        if STREET:
            if SCENE == "night":
                if "at_daytime" in prompt or "at_nighttime" in prompt:
                    pass
                else:
                    continue
        else:
            if SCENE == "night":
                if "at_daytime" in prompt or "at_nighttime" in prompt:
                    continue
                else:
                    pass
        
        if scene[:-1] in prompt:
            edited_image.append(path)
        else:
            input_image.append(path)

# print(edited_image[:10])
edited_image = sorted(edited_image, key=lambda x: x[-3])
input_image = sorted(input_image, key=lambda x: x[-3])

assert len(edited_image) == len(input_image), print(len(edited_image), len(input_image))

sampled_items = random.sample(list(zip(input_image, edited_image)), SIZE)

# Unpack the sampled items into separate lists
input_image, edited_image = zip(*sampled_items)
input_image = ["/".join(_) for _ in input_image]
edited_image = ["/".join(_) for _ in edited_image]

instruction = ["make it %s" % SCENE for _ in range(SIZE)]
assert len(edited_image) == len(input_image) == len(instruction)


print(input_image[:10])
print(edited_image[:10])
print(instruction[:10])


# Create a DataFrame with your data
data = {
    'input_image': input_image,
    'edit_prompt': instruction,
    'edited_image': edited_image,
}

df = pd.DataFrame(data)

# Save the DataFrame as a Parquet file
df.to_parquet(PARQUET_PATH)
print(PARQUET_PATH)
