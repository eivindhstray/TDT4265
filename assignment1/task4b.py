import numpy as np
import utils
import mnist
from task2a import pre_process_images

np.random.seed(1)
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

import random



def makingPrediction():

    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()


    X_train = pre_process_images(X_train)

    index = random.randint(0, X_train.shape[0])

    image = np.array([X_train[index]])

    printable = X_train[index]

    model = pickle.load(open('model3a.sav', 'rb'))

    a = model.forward(image)

    print(np.argmax(a))

    predicted = (np.argmax(a))

    image_2d = printable[:-1].reshape(28, 28)

    label = "predicted :" + str(predicted)

    plt.imshow(image_2d)
    plt.title(label)
    plt.show()


def visualModeltraining():
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    model = pickle.load(open('model3a_0,1.sav', 'rb'))
    image1 = X_train[0:10]
    frame = np.zeros((28, 28 * 10))
    j = 0
    for i in range(0, 28 * 10, 28):
        im = (model.w.T[j]) * image1[j]
        frame[:28, i:i + 28] = im[:-1].reshape(28, 28)
        j += 1
    print(frame.shape)
    plt.imshow(frame)
    plt.show()
    plt.imsave("task4b_softmax_weight.png", frame, cmap="gray")
    return frame

if __name__ == '__main__':
    visualModeltraining()