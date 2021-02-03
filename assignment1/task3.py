import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)
import pickle
from task4b import*


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    y_hat = np.array(model.forward(X))
    y_predicted_position = np.argmax(y_hat,axis=1)
    y_position = np.argmax(targets,axis = 1) 
    accuracy = np.count_nonzero(y_position == y_predicted_position)/X.shape[0]
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        Y_hat = model.forward(X_batch)
        model.backward(X_batch,Y_hat,Y_batch)
        model.w -= self.learning_rate*model.grad
        loss = cross_entropy_loss(Y_batch,Y_hat)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(Y_val, logits)

        accuracy_train = calculate_accuracy(
            X_train, Y_train, self.model)
        accuracy_val = calculate_accuracy(
            X_val, Y_val, self.model)
        return loss, accuracy_train, accuracy_val


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 20
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
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, 2.0])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.60, .95])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()
    
    #save model
    #filename = 'model3a_1.sav' #Uncomment to save a new version of the model

    # Train a model with L2 regularization (task 4b)
    '''
    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    pickle.dump(model, open(filename, 'wb'))

    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    '''

    # You can finish the rest of task 4 below this point.

    #plotting for different lambdas:
    l2_lambdas = [1, .1, .01, .001]

    val_acc = []

    weights = []

    for i in l2_lambdas:
        model = SoftmaxModel(l2_reg_lambda=i)
        #    Train model
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        val_acc.append(val_history["accuracy"])
        weights.append(model.w)

        # Plot accuracy
        
        utils.plot_loss(val_history["accuracy"], "Validation Accuracy with $\lambda$ ={}".format(i))
    plt.ylim([0.60, .95])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    # Plotting of softmax weights (Task 4b)
    #weight = visualModeltraining() - task4b.py 
    #plt.imsave("task4b_softmax_weight_2.png", weight, cmap="gray")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    #plt.savefig("task4c_l2_reg_accuracy.png")

    # Task 4d - Plotting of the l2 norm for each weight. Could be more elegant, but it's just a bar plot so frankly
    # I don't really care
    l2_lambdas = [str(1), str(0.1), str(0.01),str(0.001)]
    weights_normalized = []
    for w in weights: #Saved as an array during training above, one for each lambda.
        weights_normalized.append(np.linalg.norm(w,2))
    plt.bar(l2_lambdas[0], weights_normalized[0])
    plt.bar(l2_lambdas[1], weights_normalized[1])
    plt.bar(l2_lambdas[2], weights_normalized[2])
    plt.bar(l2_lambdas[3], weights_normalized[3])
    plt.xlabel("$\lambda$")
    plt.ylabel("L_2 norm")
    plt.savefig("task4d_l2_reg_norms.png")



