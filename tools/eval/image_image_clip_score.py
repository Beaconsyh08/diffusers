import torch
import numpy as np
from PIL import Image


# Load the CLIP model
clip_path = "/mnt/ve_share/songyuhao/generation/models/online/diffusions/base/clip-vit-base-patch16/pytorch_model.bin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# clip_model, clip_preprocess = torch.hub.load("openai/clip", "clip", device=device)
clip_model = torch.jit.load(clip_path, map_location=device).eval()
print("here")
clip_model.eval()

# Load your dataset of image pairs
image_paths_1 = [...]  # List of paths to the first images
image_paths_2 = [...]  # List of paths to the second images

# Preprocess the images
images_1 = ["/tos://haomo-public/lucas-generation/syh/train/instructpix2pix/imgs/replace_blend_reweight/night/0/0.80_0.80_2.00/a_car_driving_down_a_highway_at_daytime_with_mountains_in_the_background_and_a_bridge_in_the_foreground_with_a_car_driving_on_the_road_at_daytime.png"]
images_2 = ["/tos://haomo-public/lucas-generation/syh/train/instructpix2pix/imgs/replace_blend_reweight/night/0/0.80_0.80_2.00/a_car_driving_down_a_highway_at_nighttime_with_mountains_in_the_background_and_a_bridge_in_the_foreground_with_a_car_driving_on_the_road_at_nighttime.png"]

for path_1, path_2 in zip(image_paths_1, image_paths_2):
    image_1 = Image.open(path_1)
    image_2 = Image.open(path_2)

    image_1 = image_1.resize((224, 224))  # Resize to CLIP model input size
    image_2 = image_2.resize((224, 224))

    image_1 = clip_preprocess(image_1).unsqueeze(0).to(device)
    image_2 = clip_preprocess(image_2).unsqueeze(0).to(device)

    images_1.append(image_1)
    images_2.append(image_2)

# Encode the images
with torch.no_grad():
    image_features_1 = clip_model.encode_image(torch.cat(images_1))
    image_features_2 = clip_model.encode_image(torch.cat(images_2))

# Compute similarity scores using cosine similarity
similarity_scores = np.dot(image_features_1.cpu(), image_features_2.cpu().T)
similarity_scores = np.squeeze(similarity_scores)
print(similarity_scores)

# # Obtain ground truth scores (if available)
# ground_truth_scores = [...]

# # Calculate Spearman's rank correlation coefficient
# correlation, _ = spearmanr(similarity_scores, ground_truth_scores)

# # Print the CLIP score
# print("Image-to-Image CLIP Score (Spearman's correlation coefficient):", correlation)
