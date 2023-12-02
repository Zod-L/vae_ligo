
import torch
import cv2
import os

scale = "4.0"
root = f"/data/ywp9786/ligo/Ligo_new/best/sub_{scale}"

for dir in os.listdir(root):
    for fname in os.listdir(os.path.join(root, dir)):
        im = cv2.imread(os.path.join(root, dir, fname))
        im = im[61 : 540, 101 : 674, :]
        im = cv2.resize(im, (512, 512))
        if not os.path.exists(os.path.join(f"../gravityspy/processed/sub_{scale}", dir)):
            os.makedirs(os.path.join(f"../gravityspy/processed/sub_{scale}", dir))
        cv2.imwrite(os.path.join(f"../gravityspy/processed/sub_{scale}", dir, fname), im)
