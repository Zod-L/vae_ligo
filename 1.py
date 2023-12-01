import cv2

im = cv2.imread("checkpoints/sub_2.0/3683-1231-4-5-128-l1-l2/80.png")
print(im.min(), im.max())
