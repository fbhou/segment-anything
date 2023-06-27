import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import ConnectionPatch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

image_path1 = r"./data/230217213828581583.png"
image_path2 = r"./data/230217213829381562.png"
matching_num = 15

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
sam = sam_model_registry["default"](checkpoint=r"./notebooks/checkpoints/sam_vit_h_4b8939.pth").to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

def generate_masks(image_path: str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    return masks

def matching_sort_key(match: list):
    return match[2]

masks1 = generate_masks(image_path1)
tokens1 = []
for _, mask in enumerate(masks1):
        tokens1.append(mask["tokens"])

masks2 = generate_masks(image_path2)
tokens2 = []
for _, mask in enumerate(masks2):
        tokens2.append(mask["tokens"])

nearest = []
for idx1, embedding1 in enumerate(tokens1):
    min_idx = 0
    min_dis2 = 1000
    for idx2, embedding2 in enumerate(tokens2):
        dis2 = 0
        for i in range(len(embedding1)):
            dis2 += (embedding1[i] - embedding2[i]) * (embedding1[i] - embedding2[i])
        if dis2 < min_dis2:
            min_dis2 = dis2
            min_idx = idx2
    nearest.append((idx1, min_idx, np.sqrt(min_dis2)))
nearest.sort(key=matching_sort_key)

fig = plt.figure(figsize=(10, 5))
image1 = np.array(Image.open(image_path1))
fig1 = fig.add_subplot(211)
fig1.imshow(image1)
image2 = np.array(Image.open(image_path2))
fig2 = fig.add_subplot(212)
fig2.imshow(image2)

for i in range(matching_num):
    idx1, idx2, dis = nearest[i]

    bbox1 = masks1[idx1]["bbox"]
    x1 = bbox1[0] + bbox1[2] / 2
    y1 = bbox1[1] + bbox1[3] / 2
    xy1 = (x1, y1)

    bbox2 = masks2[idx2]["bbox"]
    x2 = bbox2[0] + bbox2[2] / 2
    y2 = bbox2[1] + bbox2[3] / 2
    xy2 = (x2, y2)

    if dis < 5.0:
        connection_color = "red"
    elif dis < 7.0:
        connection_color = "orange"
    elif dis < 9.0:
        connection_color = "yellow"
    else:
        connection_color = "black"
    fig2.add_artist(ConnectionPatch(xyA=xy1, xyB=xy2, axesA=fig1, axesB=fig2, coordsA="data", coordsB="data", color=connection_color))

plt.show()
# fig.show()
