"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        # Might we need to store something before returning?
        self.state = 1 / (1 + np.exp(-x))
        return self.state

    def derivative(self):

        # Maybe something we need later in here...
        return self.state * (1 - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.state

    def derivative(self):
        return 1 - self.state ** 2


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x * (x > 0)
        return self.state

    def derivative(self):
        return 1. * (self.state > 0)

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None
        self.labels = None
        self.loss = None
    def forward(self, x, y):

        self.logits = x
        self.labels = y

        # ...
        # softmax
        c = np.max(self.logits, axis=1).reshape(-1, 1)
        self.sm = np.exp(c + np.log(np.exp(self.logits - c))) / np.exp(c + np.log(np.sum(np.exp(self.logits - c), axis=1).reshape(-1, 1)))
        # cross entropy loss
        batch_size = self.labels.shape[0]
        
        self.loss = -1 * np.sum(self.labels * np.log(self.sm), axis=1)
        
        return self.loss

    def derivative(self):

        # self.sm might be useful here...
        grad = self.sm - self.labels
        return grad
        
        


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        self.x = x

        if eval:
            self.norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps) 
            self.out = self.norm * self.gamma + self.beta
        else:
            x_size = x.shape[0]
            self.mean = x.mean(axis=0)
            self.var = np.sum((x - self.mean) ** 2, axis=0) / x_size
            self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)
            self.out = self.gamma * self.norm + self.beta

            # update running batch statistics
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var

        return self.out

    def backward(self, delta):
        self.dbeta = np.sum(delta, axis=0)
        self.dgamma = np.sum(delta * self.norm, axis=0)
        # dnorm = self.gamma * delta
        # # dinvert_var = np.sum(dnorm * self.mean, axis=0)
        # # dsd = -1. / (self.var + self.eps) * dinvert_var
        # # dvar = 0.5 / np.sqrt(self.var + self.eps) * dsd
        batch_size = self.x.shape[0]
        # dsd = self.norm / (self.x - self.mean)
        # dnorm = self.gamma * dsd / x_size
        # dx2 = x_size * delta - self.dbeta - self.dgamma * self.norm
        # dx = dnorm * dx2
        dx1 = delta * (1.0 / np.sqrt(self.var+self.eps))
        dvar = np.sum(delta * (self.x - self.mean) * (-0.5) * ((self.var + self.eps) ** (-1.5)), axis=0)
        dmean = np.sum(-dx1, axis=0) + dvar * np.sum(-2.0*(self.x - self.mean)) / batch_size
        dx2 = dvar * (2.0 * (self.x - self.mean) / batch_size)
        dx3 = dmean / batch_size
        dx = dx1 + dx2 + dx3
        return dx

# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.randn(d0, d1)


def zeros_bias_init(d):
    return np.zeros(d)


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        # for 0 hidden layer
        # self.W = [weight_init_fn(input_size, output_size)]
        # self.dW = [snp.zeros(self.W[0].shape)]
        # self.b = [bias_init_fn(output_size)]
        # self.db = [np.zeros(self.b[0].shape)]
        layer_size = [input_size] + hiddens + [output_size]
        self.l = len(layer_size)
        self.W = [weight_init_fn(layer_size[lay_num], layer_size[lay_num+1]) for lay_num in range(self.l-1)]
        self.dW = [np.zeros(self.W[i].shape) for i in range(self.l-1)]
        self.b = [bias_init_fn(i) for i in layer_size[1:]]
        self.db = [np.zeros(self.b[i].shape) for i in range(self.l-1)]
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = [BatchNorm(fan_in=layer_size[i+1]) for i in range(num_bn_layers)]
        # Feel free to add any other attributes useful to your implementation (input, output, ...)
        self.y = [np.zeros((size, 1)) for size in layer_size]
        self.batch_size = None
        self.delta_W = [np.zeros(self.W[i].shape) for i in range(self.l-1)]
        self.delta_b = [np.zeros(self.b[i].shape) for i in range(self.l-1)]
        self.z = 0
    def forward(self, x):
        self.y[0] = x
        self.batch_size = x.shape[0]
        for i in range(self.l-1):
            self.z = np.dot(self.y[i], self.W[i]) + self.b[i]
            # for batch normalization
            if self.bn and i < self.num_bn_layers:
                self.z = self.bn_layers[i].forward(x=self.z, eval=not(self.train_mode))

            self.y[i+1] = self.activations[i].forward(self.z)

        return self.y[-1]

    def zero_grads(self):
        if self.train_mode:
            self.dW = [np.zeros(self.W[i].shape) for i in range(self.l-1)]
            self.db = [np.zeros(self.b[i].shape) for i in range(self.l-1)]
            # for batch normalization
            if self.bn:
                for i in range(self.num_bn_layers):
                    self.bn_layers[i].dgamma = np.zeros(self.bn_layers[i].dgamma.shape)
                    self.bn_layers[i].dbeta = np.zeros(self.bn_layers[i].dbeta.shape)


    def step(self):
        if self.train_mode:
            for i in range(self.l-1):
                # Momentum method
                if self.momentum != 0:
                    self.delta_W[i] = self.momentum * self.delta_W[i] - self.lr * self.dW[i]
                    self.delta_b[i] = self.momentum * self.delta_b[i] - self.lr * self.db[i]
                    self.W[i] += self.delta_W[i]
                    self.b[i] += self.delta_b[i]
                # Regular method
                else:
                    self.W[i] = self.W[i] - self.lr * self.dW[i]
                    self.b[i] = self.b[i] - self.lr * self.db[i]
        
        if self.bn:
            for i in range(self.num_bn_layers):
                self.bn_layers[i].gamma = self.bn_layers[i].gamma - self.lr * self.bn_layers[i].dgamma
                self.bn_layers[i].beta = self.bn_layers[i].beta - self.lr * self.bn_layers[i].dbeta
    
    def backward(self, labels):
        Loss = self.criterion.forward(self.y[-1], labels)
        if self.train_mode:
            dLoss = (self.criterion.derivative() * self.activations[-1].derivative())
            # test for one layer MLP
            # self.dW[0] = np.dot(self.x.T, dErr) / self.batch_size
            # self.db[0] = np.dot(np.ones(self.x.shape[0]).T, dErr) / self.batch_size
            # print(dLoss.shape)
            # print(self.y[-2])
            # self.dW[-1] = np.dot(self.y[-2].T, dLoss) / self.batch_size
            # self.db[-1] = np.dot(np.ones((1, self.y[-2].shape[0])), dLoss).reshape(-1,) / self.batch_size
            for i in range(self.l-2, -1, -1):
                
                if i < self.num_bn_layers:
                    dLoss = self.bn_layers[i].backward(dLoss)
                self.dW[i] = np.dot(self.y[i].T, dLoss) / self.batch_size
                self.db[i] = np.dot(np.ones((1, self.y[i].shape[0])), dLoss).reshape(-1,) / self.batch_size
                dLoss = (np.dot(dLoss, self.W[i].T) * self.activations[i-1].derivative())


    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...
        np.random.shuffle(idxs)
        trainingLoss = 0
        trainingError = 0
        ValidationLoss = 0
        ValidationError = 0
        train_batch_num = 0
        val_batch_num = 0
        
        for b in range(0, len(trainx), batch_size):

            # Train ...
            mlp.train()
            mlp.zero_grads()
            batch = idxs[b:b + batch_size]
            mlp.forward(trainx[batch])
            mlp.backward(trainy[batch])
            mlp.step()

            tLoss = mlp.criterion.loss

            trainingLoss += (np.mean(tLoss))
            pred = np.argmax(mlp.criterion.sm, axis=1)
            true = np.argmax(trainy[batch], axis=1)
            trainingError += (np.sum(true != pred) / batch_size)
            train_batch_num += 1

        
        for b in range(0, len(valx), batch_size):

            # Val ...
            mlp.eval()
            mlp.forward(valx[b:b + batch_size])
            mlp.backward(valy[b:b + batch_size])
            vLoss = mlp.criterion.loss
            ValidationLoss += (np.mean(vLoss))
            pred = np.argmax(mlp.criterion.sm, axis=1)
            true = np.argmax(valy[b:b + batch_size], axis=1)
            ValidationError += (np.sum(true != pred) / batch_size)
            val_batch_num += 1
        

        # Accumulate data...
        # print(trainingLoss / train_batch_num)
        training_losses.append(trainingLoss / train_batch_num)
        training_errors.append(trainingError / train_batch_num)
        validation_losses.append(ValidationLoss / val_batch_num)
        validation_errors.append(ValidationError / val_batch_num)
        
    # Cleanup ...

    for b in range(0, len(testx), batch_size):

        # Test ...
        mlp.eval()
        predicted_label = mlp.forward(testx[b:b + batch_size])
        predicted_label = np.argmax(predicted_label, axis=1)

    # Return results ...

    return (training_losses, training_errors, validation_losses, validation_errors)
