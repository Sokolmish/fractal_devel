import sys
import signal
import numpy as np

import matplotlib.image

from decompress import *
from storage import *

np.random.seed(12345)
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

if len(sys.argv) != 3:
    print(f'Usage: {sys.argv[0]} <image_path> <result_path>')
    exit(1)

save_filename = sys.argv[1]
img_filename = sys.argv[2]

cdata2 = load_compressed(save_filename)
d_img = decompress_y(cdata2)
matplotlib.image.imsave(img_filename, d_img, cmap='gray')
