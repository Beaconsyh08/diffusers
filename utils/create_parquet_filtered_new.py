import pandas as pd
import random
import os
import json
from tqdm import tqdm

SIZE = 4000
PARQUET_PATH = "/mnt/ve_share/songyuhao/generation/data/train/diffusions/parquet_v2/filtered_tgt/%s" % SIZE 
os.makedirs(PARQUET_PATH, exist_ok=True)
FILTERED_PATH = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/filtered_new/filtered_%s.txt" % SIZE
PARQUET_PATH = "%s/pcn.parquet" % PARQUET_PATH

with open (FILTERED_PATH, "r") as input_file:
    print(FILTERED_PATH)
    paths = [_.strip().split("/") for _ in input_file.readlines()]
    


# -5: refine/replce
# -4: scene
# -3: id
# -2: parameter

input_image, edited_image, instruction = [], [], []
for ind, path in tqdm(enumerate(paths)):
    mode = path[-5]
    scene = path[-4]
    id = path[-3]
    parameter = path[-2]
    prompt = path[-1]
    if scene == "night":
        if "nighttime" in prompt:
            edited_image.append(path)
        else:
            input_image.append(path)
    else:
        if scene[:-1] in prompt:
            edited_image.append(path)
        else:
            input_image.append(path)
    if ind % 2 == 0:
        instruction.append("make it %s" % scene)
    
print(len(edited_image))
print(len(input_image))
print(len(instruction))

assert len(edited_image) == len(input_image) == len(instruction) 

input_image = ["/".join(_) for _ in input_image]
edited_image = ["/".join(_) for _ in edited_image]

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
print(df)
# Save the DataFrame as a Parquet file
df.to_parquet(PARQUET_PATH)
print(PARQUET_PATH)
