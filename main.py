import numpy as np
from glob import glob
from PIL import Image
from matplotlib import pyplot as plt
from felzenszwalb_segmentation import segment
import cv2 as cv


image_files = glob('/Users/chengwensun/Library/CloudStorage/GoogleDrive-nikki1231121@gmail.com/Other computers/My Computer/Umass/FALL 2022/ECE 597IP/Project2/*.tif')

print(len(image_files))

image = np.array(Image.open(image_files[0]))
segmented_image = segment(image, 0.2, 400, 50)
plt.imshow(segmented_image.astype(np.uint8))
cv.imwrite('0.png',segmented_image)
plt.show()