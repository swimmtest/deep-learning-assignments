import torchvision
from matplotlib import pyplot as plt
import numpy as np

# display image dataset
def imgDisplay(data_loader):
    img, label = iter(data_loader).next()
    plt.figure(figsize=(16,24))
    img_grid = torchvision.utils.make_grid(img[:24])
    np_img_grid = img_grid.numpy()
    plt.imshow(np.transpose(np_img_grid, (1,2,0)))
    plt.show()