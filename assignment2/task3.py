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

    # Example created in assignment text - Comparing with and without shuffling.
    # YOU CAN DELETE EVERYTHING BELOW!

    #Task 3 below

    '''
    
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
    

    
    use_improved_weight_init = True
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


    
    use_momentum = True
    learning_rate_momentum = 0.02
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate_momentum, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_final, val_history_final= trainer_shuffle.train(
        num_epochs)

    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Improved sigmoid", npoints_to_average=10)
    utils.plot_loss(
        train_history_weights["loss"], "Improved weights", npoints_to_average=10)
    utils.plot_loss(
        train_history_final["loss"], "Using momentum", npoints_to_average=10)
    plt.ylabel("Validation Loss")
    plt.ylim([0, .4]) 
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.ylim([0.89, .99])
    utils.plot_loss(
        val_history["accuracy"], "Improved sigmoid")
    utils.plot_loss(
        val_history_weights["accuracy"], "Improved weights")
    
    utils.plot_loss(
        val_history_final["accuracy"], "Using momentum")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task3_2.png")
    plt.show()
    
    
    
    
    #############################
    #Task 4 a and b below        #
    #############################
    #Model from task 3
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True




    learning_rate_momentum = 0.02
    neurons_per_layer = [32,10]
    model_32 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_32, learning_rate_momentum, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_32, val_history_32= trainer_shuffle.train(
        num_epochs)
        

    neurons_per_layer = [128,10]
    model_128 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_128, learning_rate_momentum, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_128, val_history_128= trainer_shuffle.train(
        num_epochs)

    
    neurons_per_layer = [64,10]
    model_64 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_64, learning_rate_momentum, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_64, val_history_64= trainer_shuffle.train(
        num_epochs)

    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0.00, 0.4])
    utils.plot_loss(
        train_history_32["loss"], "32 hidden units", npoints_to_average=10)
    utils.plot_loss(
        train_history_128["loss"], "128 hidden units", npoints_to_average=10)
    utils.plot_loss(
        train_history_64["loss"], "64 hidden units", npoints_to_average=10)
    plt.ylabel("Training Loss")
    
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.89, .99])
    utils.plot_loss(
        val_history_128["accuracy"], "128 hidden units")
    utils.plot_loss(
        val_history_64["accuracy"], "64 hidden units")
    utils.plot_loss(
        val_history_32["accuracy"], "32 hidden units")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4_a_b.png")
    plt.show()

    
    
    ###################
    # Task 4e)        #
    ###################
    '''
    learning_rate_momentum = 0.02
    neurons_per_layer = [64,64,64,64,64,64,64,64,64,64,10]
    model_64_layers = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_64_layers, learning_rate_momentum, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_64_layers, val_history_64_layers= trainer_shuffle.train(
        num_epochs = 30)

    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    #plt.ylim([0.00, 1.00])
    utils.plot_loss(
        train_history_64_layers["loss"], "10 hidden layers", npoints_to_average=10)
    plt.ylabel("Training Loss")
    utils.plot_loss(val_history_64_layers["loss"],"Validation Loss")
    plt.ylabel("Validation Loss")
    
    plt.legend()

    plt.subplot(1, 2, 2)
    #plt.ylim([0.50, .99])
    utils.plot_loss(
        val_history_64_layers["accuracy"], "10 hidden layers")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4_e.png")
    plt.show()
    

    