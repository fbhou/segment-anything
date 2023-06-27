import cv2
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sam = sam_model_registry["default"](checkpoint=r"./notebooks/checkpoints/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

def generate_mask_embeddings(image_path: str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    tokens = []
    for _, mask in enumerate(masks):
        tokens.append(mask["tokens"])
    return tokens

tokens1 = generate_mask_embeddings(r"./data/230217213828581583.png")
tokens2 = generate_mask_embeddings(r"./data/230217213829381562.png")
print(len(tokens1), len(tokens2))
print(len(tokens1[0]), len(tokens2[0]))

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
    nearest.append((min_idx, np.sqrt(min_dis2)))
for idx, tup in enumerate(nearest):
    print(idx, tup)
