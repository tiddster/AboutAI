import numpy as np
import skimage
import matplotlib.pyplot as plt

res = np.zeros((256, 256), dtype='uint8')
for i in range(256):
    if i < 64 or 2 * 64 <= i < 3 * 64:
        temp = np.linspace(0, 255, 256)
    else:
        temp = np.linspace(255, 0, 256)
    res[i][:] = temp[:]

print(res)
skimage.io.imshow(res)
plt.show()
