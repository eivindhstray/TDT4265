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
    plt.imshow(frame)
    plt.show()
    plt.imsave("task4b_softmax_weight.png", frame, cmap="gray")
    return frame

def makeAccimage():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 2
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0.01
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model

    #plotting for different lambdas:
    l2_lambdas = [1, .1, .01, .001]

    for i in l2_lambdas:
        model = SoftmaxModel(l2_reg_lambda=1)
        #    Train model
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)

    # Plot accuracy
    plt.ylim([0.60, .95])
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy with $\lambda$ ={}".format(i))

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    makeAccimage()