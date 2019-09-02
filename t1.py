import NNF_pytorch as NNF
from NeuralNetwork import NeuralNetwork
import sys

# Process the data.

X, Y, y = NNF.dataSet([sys.argv[1], 0], 10, True)
# X, Y, y = NNF.dataSet(["mnist.csv", 0], 10, True)
X_train, Y_train, y_train, X_test, Y_test, y_test = X[:, :60000], Y[:, :60000], y[:60000], X[:, 60000:], Y[:, 60000:], y[60000:]

# Set, train and save NN1.
list1 = sys.argv[2].split(',')
list1 = list(map(float, list1))
list2 = sys.argv[4].split(',')
list2 = list(map(float, list2))
list3 = sys.argv[5].split(',')
list3 = list(map(float, list3))
NN1 = NeuralNetwork(list1, float(sys.argv[3]), list2, list3, float(sys.argv[6]), float(sys.argv[7]))
# NN1 = NeuralNetwork([784, 1, 56, 2, 56, 2, 10, 2], 1, [1, 1, 0.8], [1, 0.12], lam=0, p=1)
NN1.train(X_train, Y_train)
NN1.save("NN1")

## RUN ONE OF THE FOLLOWING COMMANDS ON THE TERMINAL.
# TO RUN IT LOCALLY:
# python t1.py mnist.csv 784,1,56,2,56,2,10,2 1 1,10000,0.8 1,0.12 0 1
# TO RUN IT ON THE CLOUD VIA SPELL:
# spell run --machine-type V100 --mount uploads/Dataset/mnist.csv --pip tqdm python t1.py mnist.csv 784,1,56,2,56,2,10,2 1 1,10000,0.8 1,0.12 0 1
