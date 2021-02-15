import numpy as np
import utils
import typing
np.random.seed(1)

def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)

    mean = np.mean(X)
    std = np.std(X)

    X = (X-mean)/std
    
    X = np.hstack((X, np.ones((X.shape[0], 1))))

    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    N = targets.shape[0]
    C_sum = 0
    for i in range(targets.shape[0]):
        for y, y_hat in zip(targets[i], outputs[i]):
            C_sum += -(y * np.log(y_hat))
    C = C_sum / N
    assert targets.shape == outputs.shape, \
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    return C


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []

        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if (self.use_improved_weight_init):
                fan_in = 1/np.sqrt(prev)
                mean = 0
                w = np.random.normal(mean, fan_in, w_shape)
            else:
                w = np.random.uniform(-1,1,w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For peforming the backward pass, you can save intermediate activations in varialbes in the forward pass.
        # such as self.hidden_layer_ouput = ...
        self.zs= []
        self.activations = [X]

        layer_output = X

        
        for weights in self.ws:
            z = layer_output.dot(weights)
            if(self.use_improved_sigmoid):
                layer_output = np.tanh(z) 
            else:
                layer_output = 1/(1+np.exp(-z))
            self.zs.append(z)
            self.activations.append(layer_output)

        softmax_layer = np.exp(z)/np.transpose(np.array([np.sum(np.exp(z),axis=1)]))

        return softmax_layer

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)

        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = [np.zeros_like(weight) for weight in self.ws]

        N = X.shape[0]



        if(self.use_improved_sigmoid):
            sigmoid = lambda a: np.tanh(a)
            sigmoid_prime = lambda b: 1-sigmoid(b)**2
        else:
            sigmoid = lambda a : 1/(1+np.exp(-a))
            sigmoid_prime = lambda b : sigmoid(b)*(1-sigmoid(b))
        
        '''
        The last layer is computed by itself.
        '''
        dk = -(targets-outputs) 
        delta_k = 1/N*(self.activations[-2].T@dk)
        self.grads[-1] = delta_k
        dj = dk
        '''
        If there are any hidden layers, these will be computed by the loop below.
        '''
        for layer in range(1,len(self.grads)):
            sp = sigmoid_prime(self.zs[-layer-1])
            dj = sp*(dj@self.ws[-layer].T)
            grad_j = (1/N)*(self.activations[-layer-2].T@dj)
            self.grads[-layer-1] = grad_j
    
        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
        


    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    num_examples = Y.shape[0]
    encoded_array = np.zeros((num_examples, num_classes))
    for i, target in enumerate(Y):
        encoded_array[i][target] = 1

    return encoded_array


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in (enumerate(model.ws)):
        print("layer idx")
        for i in range(w.shape[0]):
            print(i," ", w.shape[0])
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
            
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"

        
if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64,10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
