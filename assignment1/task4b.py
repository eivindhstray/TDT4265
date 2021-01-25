import numpy as np
import utils
import mnist
from task2a import pre_process_images
np.random.seed(1)
from tqdm import tqdm
import matplotlib.pyplot as plt

image1 = mnist.load()[0]

image1 = image1[3]

image1 = image1.reshape(28,28)

print(image1)
plt.imshow(image1)
plt.show()


