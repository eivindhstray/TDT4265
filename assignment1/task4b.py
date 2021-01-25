import numpy as np
import utils
import mnist
from task2a import pre_process_images
np.random.seed(1)
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

import random

X_train, Y_train, X_val, Y_val = utils.load_full_mnist()

X_train = pre_process_images(X_train)

index = random.randint(0,X_train.shape[0])

image = np.array([X_train[index]])

printable=X_train[index]


model = pickle.load(open('model3a.sav', 'rb'))

a = model.forward(image)

print(np.argmax(a))

predicted = (np.argmax(a))

image_2d = printable[:-1].reshape(28,28)

label = "predicted :" + str(predicted)



plt.imshow(image_2d)
plt.title(label)
plt.show()



model = pickle.load(open('model3a_1.sav', 'rb'))
image1 = X_train[0]
im = (model.w.T[0])*image1
im = im[:-1].reshape(28,28)
plt.imshow(im)
plt.show()

#plt.imshow(img)




