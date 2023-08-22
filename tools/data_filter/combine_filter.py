from tqdm import tqdm
import json
import math

scenes = ["refine_blend_reweight_snowy", "replace_blend_reweight_night", "replace_blend_reweight_foggy", "replace_blend_reweight_rainy"]
# scenes = ["refine_blend_reweight_snowy"]

paras = ["0.60_0.60_2.00", "0.70_0.70_2.00","0.80_0.80_2.00"]
ori_json_root = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/ori"
img_root = "/tos://haomo-public/lucas-generation/syh/train/prompt2prompt_controlnet_0810/imgs"
P2P_PATH = "/tos://haomo-public/lucas-generation/syh/train/prompt2prompt_controlnet_0810/index.txt"
topK = 4000
result_path = "/mnt/ve_share/songyuhao/generation/data/filtered_p2p_cn/filtered_new/filtered_%s.txt" % topK

with open (P2P_PATH, "r") as input_file:
    print(P2P_PATH)
    # paths = [_.strip().split("/") for _ in input_file.readlines()]
    paths = [_.strip() for _ in input_file.readlines()]
    

combine_res = []
for scene in scenes:
    print(scene)
    res_dic = []
    for para in paras:
        ori_json = "%s/%s_%s_all.json" % (ori_json_root, scene, para) 
        with open(ori_json) as json_res:
            res = json.load(json_res)
            for i in tqdm(range(len(res))):
                id_ = res[i]["id"]
                image_image_sim = res[i]["image_image_sim"]
                image_caption_sim_1 = res[i]["image_caption_sim_1"]
                image_caption_sim_2 = res[i]["image_caption_sim_2"]
                directional_sim = res[i]["directional_sim"]
                if image_image_sim < 0.75 or image_caption_sim_1 < 0.2 or image_caption_sim_2 < 0.2 or directional_sim < 0.2 or math.isnan(directional_sim):
                    pass
                else:
                    name = "%s/%s/%s/%d/%s" % (img_root, "_".join(scene.split("_")[:-1]), scene.split("_")[-1], id_, para)
                    res_dic.append({"path": name, "score": directional_sim, "id": id_})
    res_sorted = sorted(res_dic, key=lambda x: x["score"], reverse=True)
    print(len(res_sorted))
    print(res_sorted[:10])
    print(res_sorted[-10:])
    
    # print("Pass Ratio: ", (round(len(res_sorted)/ len(res) / len(paras), 4)) * 100)
    
    unique_dicts = []
    unique_ids = set()

    # Iterate through the list of dictionaries
    for d in tqdm(res_sorted[:topK*len(paras)]):
        if d['id'] not in unique_ids:
            unique_dicts.append(d)
            unique_ids.add(d['id'])
    print(len(unique_dicts))
    unique_dicts = unique_dicts[:topK]
    print(unique_dicts[:10])
    print(unique_dicts[-10:])
    combine_res += [_["path"] for _ in unique_dicts]
    print(len(combine_res))
    
real_res = []
for full_path in tqdm(paths):
    for sub_path in combine_res:
        if sub_path in full_path:
            real_res.append(full_path)
with open(result_path, "w") as output_file:
    for real_res_path in real_res:
        output_file.writelines(real_res_path + "\n")
print(result_path)
