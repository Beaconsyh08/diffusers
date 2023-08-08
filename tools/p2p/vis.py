import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

json_path = "/mnt/ve_share/songyuhao/generation/data/p2p_exp/exp2_ns.json"
save_path_excel = "/mnt/ve_share/songyuhao/generation/data/p2p_exp/res/directional_sim.xlsx"
save_path_hotmap = "/mnt/ve_share/songyuhao/generation/data/p2p_exp/res/directional_sim.png"




with open(json_path) as input_file:
    input_info = json.load(input_file)
    
directional_sims = [_["directional_sim"] for _ in input_info]
directional_sims = [directional_sims[i:i+10] for i in range(0, len(directional_sims), 10)]

df = pd.DataFrame(directional_sims, columns=[round(0.4*(i+1), 1) for i in range(10)])
cross_steps = [round(0.1*(i+1), 1) for i in range(9)]
df["cross_steps"] = cross_steps
df.set_index("cross_steps", inplace=True)
print(df)
df.to_excel(save_path_excel, index=True, header=True)
print(save_path_excel)

# plt.imshow(directional_sims, annot=True, fmt=".2f", cmap='Blues', interpolation='nearest')
sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu",)
# plt.colorbar()  # Add a colorbar for reference
plt.savefig(save_path_hotmap)
print(save_path_hotmap)
