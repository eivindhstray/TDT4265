import numpy as np
import utils
import mnist
from task2a import pre_process_images
from trainer import BaseTrainer
from task2 import LogisticTrainer
from task2a import BinaryModel
from task3 import*

np.random.seed(1)
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

import random



def predictAndDisplay():  #This is just a test function that predicts a random handwritten number from the dataset.

    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()


    X_train = pre_process_images(X_train)

    index = random.randint(0, X_train.shape[0])

    image = np.array([X_train[index]])

    printable = X_train[index]

    model = pickle.load(open('model3a.sav', 'rb'))

    a = model.forward(image)


    predicted = (np.argmax(a))

    image_2d = printable[:-1].reshape(28, 28)

    label = "predicted :" + str(predicted)

    plt.imshow(image_2d)
    plt.title(label)
    plt.show()


def visualModeltraining(): #This function generates an image from the weights
    
    model = pickle.load(open('model_lambda_1.sav', 'rb')) #Change this for different models.
    frame = np.zeros((28, 28 * 10))
    j = 0
    for i in range(0, 28 * 10, 28):
        im = (model.w.T[j][:784])
        frame[:28, i:i + 28] = im.reshape(28, 28)
        j += 1
    plt.imshow(frame)
    plt.imsave("task4b_softmax_weight_1.png", frame, cmap="gray")
    plt.show()
    return frame

if __name__ == '__main__':
    visualModeltraining()