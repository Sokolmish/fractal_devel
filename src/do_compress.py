import sys
import signal
import numpy as np
import matplotlib.image as mpimg

from compress import *
from storage import *


# TODO: duplicate
def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 3:
        return np.mean(img[:, :, :2], 2)
        # return rgb2ycbcr(img)[:, :, 0]
    return img


# TODO: duplicate
def remove_alpha(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3 and img.shape[2] == 4:
        return img[:, :, :3]
    return img


np.random.seed(12345)
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

if len(sys.argv) != 3:
    print(f'Usage: {sys.argv[0]} <image_path> <result_path>')
    exit(1)

img_filename = sys.argv[1]
save_filename = sys.argv[2]

img = mpimg.imread(img_filename)

img = remove_alpha(img)
img = to_grayscale(img)
img = img.astype(np.float32)

cdata = compress_y(img)
store_compressed(cdata, save_filename)
