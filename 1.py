import cv2
import os

for root, dirs, files in os.walk("/home/otn7723/gravityspy/processed"):
    if len(dirs) > 0:
        print(root)
        print(dirs)
        print(files)
        print()
