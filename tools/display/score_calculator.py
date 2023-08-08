from collections import defaultdict
import os
import pandas as pd

root_path = "/mnt/ve_share/songyuhao/generation/records/txt"
save_path = "/mnt/ve_share/songyuhao/generation/records/xlsx"

selected_user = "syh"

def get_file_paths(folder_path):
    file_paths = []
    for root, directories, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)
    return file_paths

# Call the function to get the file paths
paths = get_file_paths(root_path)

for path in paths:
    correct_dict = defaultdict(int)
    correct_dict_2 = defaultdict(int)
    all_dict = defaultdict(int)
    
    with open(path) as result_file:
        ress = [_.strip() for _ in result_file.readlines()]
    
    model = path.split("/")[-1][:-4]
    
    ress = sorted(ress, key=lambda x:x.split("@")[1].split("_")[-1])
    for res in ress:
        res_split = res.split("@")
        user = res_split[0]
        if user == selected_user:
            if len(res_split) == 4: 
                scene = res_split[1].split("_")[-1].capitalize()
                # if model == "INS-Base":
                #     print(scene)
                if scene not in ["Backlight"]:
                    if int(res_split[2]) > int(res_split[3]):
                        continue
                    correct_dict[scene] += int(res_split[2])
                    all_dict[scene] += int(res_split[3])
            elif len(res_split) == 5:
                print(res_split)
                scene = res_split[1].split("_")[-1].capitalize()
                # if model == "INS-Base":
                #     print(scene)
                if scene not in ["Backlight"]:
                    if int(res_split[2]) > int(res_split[4]):
                        continue
                    correct_dict[scene] += int(res_split[2])
                    correct_dict_2[scene] += int(res_split[3])
                    all_dict[scene] += int(res_split[4])
    scenes = list(correct_dict.keys())
    
    res_dict = dict()
    if len(res_split) == 4:
        for scene in scenes:
            res_dict[scene] = round(correct_dict[scene] / all_dict[scene], 4)
        if sum((list(all_dict.values()))) != 0:
            res_dict["Average"] = round(sum(list(correct_dict.values())) / sum((list(all_dict.values()))), 4)
            
        df = pd.DataFrame(res_dict, index=["通过率"])

    elif len(res_split) == 5:
        for scene in scenes:
            res_dict[scene] = round(correct_dict[scene] / all_dict[scene], 4)
        if sum((list(all_dict.values()))) != 0:
            res_dict["Average"] = round(sum(list(correct_dict.values())) / sum((list(all_dict.values()))), 4)
        df1 = pd.DataFrame(res_dict, index=["标注可复用率"])
    
        for scene in scenes:
            res_dict[scene] = round(correct_dict_2[scene] / all_dict[scene], 4)
        if sum((list(all_dict.values()))) != 0:
            res_dict["Average"] = round(sum(list(correct_dict_2.values())) / sum((list(all_dict.values()))), 4)
        df2 = pd.DataFrame(res_dict, index=["场景成功翻译率"])
        
        df = pd.concat([df1, df2], axis=0)
    
    print(df)
    print(model)
    print(res_dict)
    output_path = "%s/%s_%s.xlsx" % (save_path, model, selected_user)
    df.to_excel(output_path)
    print(output_path)