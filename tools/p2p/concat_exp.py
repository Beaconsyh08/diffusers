import os
import cv2

imgs_root = "/mnt/ve_share/songyuhao/generation/data/p2p_cn/imgs/exp2"

image_1 = "%s/ori.png" % imgs_root
for root, folders, files in os.walk(imgs_root):
    files.remove("ori.png")
    image_2s = [os.path.join(root, f) for f in files]
    
image_2s = [cv2.resize(cv2.imread(_), (256, 256)) for _ in image_2s]
image_2s = [image_2s[i:i+10] for i in range(0, len(image_2s), 10)]

concat_img = cv2.vconcat([cv2.hconcat(list_h) for list_h in image_2s])
cv2.imwrite("/mnt/ve_share/songyuhao/generation/data/p2p_exp/concat.png", concat_img)

