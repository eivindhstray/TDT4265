import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer

from trainer import BaseTrainer
import numpy as np
np.random.seed(0)

if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Example created in assignment text - Comparing with and without shuffling.
    # YOU CAN DELETE EVERYTHING BELOW!
    
    shuffle_data = False
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_weights, val_history_weights= trainer_shuffle.train(
        num_epochs)
    shuffle_data = True

    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Improved sigmoid", npoints_to_average=10)
    utils.plot_loss(
        train_history_weights["loss"], "Improved weights", npoints_to_average=10)
    plt.ylim([0, .4]) 
    plt.subplot(1, 2, 2)
    plt.ylim([0.89, .95])
    utils.plot_loss(val_history["accuracy"], "Improves sigmoid")
    utils.plot_loss(
        val_history_weights["accuracy"], "Improved weights")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task3_2.png")
    plt.show()
    
    
