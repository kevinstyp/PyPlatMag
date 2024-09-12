import numpy as np

class StepDecay():
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10, first_extra=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery
        self.first_extra = -first_extra

    def __call__(self, epoch, lr):
        #print("someone called StepDecay")
        # compute the learning rate for the current epoch
        # first_extra adds a negative value in the beginning, thus pro-longing the very first training phase
        # Therefore, also the max() with 0.0 to not receive negative values
        exp = np.max([np.floor((self.first_extra + epoch) / self.dropEvery), 0.0])
        alpha = self.initAlpha * (self.factor ** exp)
        # return the learning rate
        #print("lr last: ", lr)
        #print("lr now: ", alpha)
        if np.abs(lr - alpha) > 1e-09:
            print("LRS Changed learning rate to: ", alpha)
        #print("float-alpha: ", float(alpha))
        return float(alpha)