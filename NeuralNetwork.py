import NNF_pytorch as NNF  # Functions that are usful for the developer, but not for the common user are not part of the NeuralNetwork class. Instead, they are written in this separate module.
import torch  # This is optional. Import numpy instead of torch if using NNF_numpy instead of NNF_pytorch.
import pickle  # To save instances of this class.


class NeuralNetwork():
    # Set the hyperparameters of the neural network.
    def __init__(self, archi, cf, opt, init, lam=0, p=1):
        # Hyperparameters.
        self.layers_size = archi[0::2]
        self.layers_af = archi[1::2]
        self.opt = opt
        self.cf = cf
        self.lam = lam
        self.p = p
        # Parameters.
        self.lot_EWM, self.lot_EBV = NNF.f1(init, self.layers_size)

    # Train the neural network.
    def train(self, X, Y):
        self.lot_EWM, self.lot_EBV, J_history = NNF.opt(self.cf, self.opt, X, Y, self.lot_EWM, self.lot_EBV, self.layers_size, self.layers_af, self.lam, self.p)
        return self.lot_EWM, self.lot_EBV, J_history

    # Use the neural network to make prediction. It's obviously only going to work well if the neural network is trained.
    def prediction(self, X):
        A = X
        for j in range(len(self.layers_size) - 1):
            Z = self.lot_EWM[j] @ A + self.lot_EBV[j]
            A = NNF.af(self.layers_af, Z, j + 1)
        prediction = A.argmax(0)
        prediction = prediction.reshape(len(prediction), 1)
        return prediction

    # Evaluate the accuracy of the neural network's prediction.
    def simpleEval(self, X, y):
        prediction = self.prediction(X)
        simpleEval = torch.eq(y, prediction.float()).float()
        simpleEval = (sum(simpleEval) / len(y)).item() * 100
        return simpleEval

    # def trueEval(self, data):
    #     return trueEval

    def save(self, fileName):
        po = open(fileName, "wb")
        pickle.dump(self, po)
        po.close()
