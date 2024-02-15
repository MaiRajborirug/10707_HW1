"""
Spring 2024, 10-707
Homework 1
Problem 5: MLP
TAs in charge: 
    Jiatai Li (jiatail)
    Kaiwen Geng (kgeng)
    Torin Kovach (tkovach)

IMPORTANT:
    DO NOT change any function signatures

Jan 2024
"""

import numpy as np


def random_weight_init(indim, outdim):
    b = np.sqrt(6) / np.sqrt(indim + outdim)
    return np.random.uniform(-b, b, (indim, outdim))


def zeros_bias_init(outdim):
    return np.zeros((outdim, 1))


class Transform:
    """
    This is the base class. You do not need to change anything.

    Please read the comments in this class carefully.
    """

    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        In this function, we accumulate the gradient values instead of assigning
        the gradient values. This allows us to call forward and backward multiple
        times while only update parameters once.
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


class ReLU(Transform):
    """
    Implement this class
    """

    def __init__(self):
        Transform.__init__(self)
        self.x = None

    def forward(self, x, train=True):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_wrt_out):
        # pass only the gradients of the positive values
        dx = np.array(grad_wrt_out, copy=True)
        dx[self.x <= 0] = 0
        return dx


class LinearMap(Transform):
    """
    Implement this class
    Please use *_init() functions given at the beginning of this file
    """

    def __init__(self, indim, outdim, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        """
        indim: input dimension
        outdim: output dimension
        alpha: parameter for momentum updates
        lr: learning rate
        """
        self.alpha = alpha
        self.lr = lr
        self.W = random_weight_init(indim, outdim)
        self.b = zeros_bias_init(outdim)
        self.vW = np.zeros((indim, outdim)) # also initiate momentum features
        self.vb = np.zeros((outdim, 1))
        

    def forward(self, x):
        """
        x shape (batch_size, indim)
        return shape (batch_size, outdim)
        """
        self.x = x
        return x @ self.W + self.b.T


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, outdim)
        return shape (batch_size, indim)
        Your backward call should Accumulate gradients.
        """
        # HW 3.3 dy_k/dw_ij = delta_ik x_j
        # dL/dy = grad_wrt_out
        # exp: gradW = dL/dw (indim, outdim) = dL/dy * dy/dw = grad_wrt_out(batch_size, outdim) . x(batch_size, indim)
        
        # HW 3.3 dy_i/db_j = delta_ij  
        # exp: gradb = dL/db (outdim,1) = dL/dy * dy/db = grad_wrt_out(batch_size, outdim) . delta(outdim, 1)
        self.gradW = self.x.T @ grad_wrt_out
        self.gradb = np.sum(grad_wrt_out, axis=0, keepdims=True).T
        self.gradX = grad_wrt_out @ self.W.T
        return self.gradX # for previous layer

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters

        Make sure your gradient step takes into account momentum.
        Use alpha as the momentum parameter.
        """
        # v = alpha * v - lr * grad
        # record momentum
        self.vW = self.alpha * self.vW - self.lr * self.gradW
        self.vb = self.alpha * self.vb - self.lr * self.gradb
        self.W += self.vW
        self.b += self.vb

    def zerograd(self):
        # reset parameters
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def getW(self):
        # return weights
        return self.W

    def getb(self):
        # return bias
        return self.b

    def loadparams(self, w, b):
        # Used for Autograder. Do not change.
        self.W, self.b = w, b


class SoftmaxCrossEntropyLoss:
    """
    Implement this class
    """

    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are true labels of given inputs
        labels are one-hot encoded
        logits and labels are in the shape of (batch_size, num_classes)
        returns loss as scalar
        (your loss should be a mean value on batch_size)
        """
        self.logits = logits
        self.labels = labels
        self.batch_size = labels.shape[0]
        # make sure the softmax is stable with minus max
        exp_ = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.softmax = exp_ / np.sum(exp_, axis=1, keepdims=True)
        # prevent log(0) = nan
        loss = -np.sum(labels * np.log(self.softmax + 1e-10)) / self.batch_size
        return loss

    def backward(self):
        """
        return shape (batch_size, num_classes)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        # pb 3.1: dL/do = p-y
        dpdo = (self.softmax - self.labels) / self.batch_size
        return dpdo

    def getAccu(self):
        """
        return accuracy here (as you wish)
        This part is not autograded.
        """
        pred = np.argmax(self.softmax, axis=1)
        labels = np.argmax(self.labels, axis=1)
        accuracy = np.mean(pred == labels)
        return accuracy


class SingleLayerMLP(Transform):
    """
    Implement this class
    """

    def __init__(self, indim, outdim, hiddenlayer=100, alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.linear1 = LinearMap(indim, hiddenlayer, alpha, lr)
        self.relu = ReLU()
        self.linear2 = LinearMap(hiddenlayer, outdim, alpha, lr)

    def forward(self, x, train=True):
        """
        x shape (batch_size, indim)
        return shape (batch_size, outdim)
        """
        x = self.linear1.forward(x)
        x = self.relu.forward(x)
        x = self.linear2.forward(x)
        return x

    def backward(self, grad_wrt_out):
        grad_wrt_out = self.linear2.backward(grad_wrt_out)
        grad_wrt_out = self.relu.backward(grad_wrt_out)
        grad_wrt_out = self.linear1.backward(grad_wrt_out)
        return grad_wrt_out

    def step(self):
        """"update linear1 and linear2 parameters"""
        self.linear1.step()
        self.linear2.step()

    def zerograd(self):
        self.linear1.zerograd()
        self.linear2.zerograd()

    def loadparams(self, Ws, bs):
        """
        use LinearMap.loadparams() to implement this
        Ws is a list, whose element is weights array of a layer, first layer first
        bs for bias similarly
        e.g., Ws may be [LinearMap1.W, LinearMap2.W]
        Used for autograder.
        """
        self.linear1.loadparams(Ws[0], bs[0])
        self.linear2.loadparams(Ws[1], bs[1])

    def getWs(self):
        """
        Return the weights for each layer
        You need to implement this.
        Return weights for first layer then second and so on...
        """
        return [self.linear1.getW(), self.linear2.getW()]

    def getbs(self):
        """
        Return the biases for each layer
        You need to implement this.
        Return bias for first layer then second and so on...
        """
        return [self.linear1.getb(), self.linear2.getb()]


class TwoLayerMLP(Transform):
    """
    Implement this class
    Everything similar to SingleLayerMLP
    """

    def __init__(self, inp, outp, hiddenlayers=[100, 100], alpha=0.1, lr=0.01):
        Transform.__init__(self)
        self.linear1 = LinearMap(inp, hiddenlayers[0], alpha, lr)
        self.relu1 = ReLU()
        self.linear2 = LinearMap(hiddenlayers[0], hiddenlayers[1], alpha, lr)
        self.relu2 = ReLU()
        self.linear3 = LinearMap(hiddenlayers[1], outp, alpha, lr)
        

    def forward(self, x, train=True):
        x = self.linear1.forward(x)
        x = self.relu1.forward(x)
        x = self.linear2.forward(x)
        x = self.relu2.forward(x)
        x = self.linear3.forward(x)
        return x

    def backward(self, grad_wrt_out):
        grad_wrt_out = self.linear3.backward(grad_wrt_out)
        grad_wrt_out = self.relu2.backward(grad_wrt_out)
        grad_wrt_out = self.linear2.backward(grad_wrt_out)
        grad_wrt_out = self.relu1.backward(grad_wrt_out)
        grad_wrt_out = self.linear1.backward(grad_wrt_out)
        return grad_wrt_out

    def step(self):
        self.linear1.step()
        self.linear2.step()
        self.linear3.step()

    def zerograd(self):
        self.linear1.zerograd()
        self.linear2.zerograd()
        self.linear3.zerograd()

    def loadparams(self, Ws, bs):
        self.linear1.loadparams(Ws[0], bs[0])
        self.linear2.loadparams(Ws[1], bs[1])
        self.linear3.loadparams(Ws[2], bs[2])

    def getWs(self):
        return [self.linear1.getW(), self.linear2.getW(), self.linear3.getW()]

    def getbs(self):
        return [self.linear1.getb(), self.linear2.getb(), self.linear3.getb()]


class Dropout(Transform):
    """
    Implement this class
    """

    def __init__(self, p=0.5):
        Transform.__init__(self)
        """
        p is the Dropout probability
        """
        self.p = p

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):
        """
        Get and apply a mask generated from np.random.binomial during training
        Scale your output accordingly during testing
        """
        if train:
            size = int(x.shape[0]*x.shape[1])
            self.mask = np.random.binomial(1, self.p, size).reshape(x.shape)
            return x * self.mask
        else:
            return x * self.p
        
        

    def backward(self, grad_wrt_out):
        """
        This method is only called during trianing.
        """
        return grad_wrt_out * self.mask


class BatchNorm(Transform):
    """
    Implement this class
    """

    def __init__(self, indim, alpha=0.9, lr=0.01, mm=0.9):
        Transform.__init__(self)
        """
        You shouldn't need to edit anything in init
        """
        self.alpha = alpha  # parameter for running average of mean and variance
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None
        self.lr = lr
        self.mm = mm  # parameter for updating gamma and beta
        """
        The following attributes will be tested
        """
        self.var = np.ones((1, indim))
        self.mean = np.zeros((1, indim))

        self.gamma = np.ones((1, indim))
        self.beta = np.zeros((1, indim))
        """
        gradient parameters
        """
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

        """
        momentum parameters
        """
        self.mgamma = np.zeros_like(self.gamma)
        self.mbeta = np.zeros_like(self.beta)

        """
        inference parameters
        """
        self.running_mean = np.zeros((1, indim))
        self.running_var = np.ones((1, indim))

    def __call__(self, x, train=True):
        return self.forward(x, train)

    def forward(self, x, train=True):
        """
        x shape (batch_size, indim)
        return shape (batch_size, indim)
        Please use batch mean and variance to update running averages during training,
        and use the running averages to normalize input during testing.
        """
        if train:
            # batch mean and variance
            self.batch_mean  = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)
            
            # update running averages using momentum
            self.mean = self.alpha * self.mean + (1 - self.alpha) * self.batch_mean
            self.var = self.alpha * self.var + (1 - self.alpha) * self.batch_var
            
            self.x = x
            self.norm = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)  # use batch mean and var
            out = self.gamma * self.norm + self.beta
        
        else:
            # use running averages
            self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)  # use batch mean and var
            out = self.gamma * self.norm + self.beta

        return out
            

    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (batch_size, indim)
        return shape (batch_size, indim)
        """
        m = grad_wrt_out.shape[0]        
        self.norm = grad_wrt_out * self.gamma
        x_mu = self.x - self.batch_mean
        std_inv = 1. / np.sqrt(self.batch_var + self.eps)
        
        dvar = np.sum(self.norm * x_mu, axis=0, keepdims=True) * -0.5 * std_inv ** 3
        dmean = np.sum(self.norm * -std_inv + self.eps, axis=0, keepdims=True) + dvar * np.mean(-2 * x_mu, axis=0, keepdims=True)
        dx = (self.norm * std_inv) + (dvar * 2 * x_mu / m) + (dmean / m)
        
        self.dgamma = np.sum(grad_wrt_out * self.norm, axis=0, keepdims=True) # (1, indim)
        self.dbeta = np.sum(grad_wrt_out, axis=0, keepdims=True) # (1, indim)

        return dx

    def step(self):
        """
        apply gradients calculated by backward() to update the parameters
        Make sure your gradient step takes into account momentum.
        Use mm as the momentum parameter.
        """
        self.mgamma = self.mm * self.mgamma - self.lr * self.dgamma
        self.mbeta = self.mm * self.mbeta - self.lr * self.dbeta
        self.gamma += self.mgamma
        self.beta += self.mbeta

    def zerograd(self):
        # reset parameters
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def getgamma(self):
        # return gamma
        return self.gamma

    def getbeta(self):
        # return beta
        return self.beta

    def loadparams(self, gamma, beta):
        # Used for Autograder. Do not change.
        self.gamma, self.beta = gamma, beta


def labels2onehot(labels):
    return np.eye(np.max(labels) + 1)[labels].astype(np.float32)


if __name__ == "__main__":
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    import pickle

    # change this to where you downloaded the file,
    # usually ends with 'cifar10-subset.pkl'
    CIFAR_FILENAME = "../cifar10-subset.pkl"
    with open(CIFAR_FILENAME, "rb") as f:
        data = pickle.load(f)

    # preprocess
    trainX = data["trainX"] / 255.0
    trainy = labels2onehot(data["trainy"])
    testX = data["testX"] / 255.0
    testy = labels2onehot(data["testy"])
    
    with open("tests.pk", "rb") as f:
        tests = pickle.load(f)
    
    # python -m unittest tests.TestSingleLayerMLP
    

    
    
    