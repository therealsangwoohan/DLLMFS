import NNF_pytorch as NNF


NN1 = NNF.load("NN1")

# Process the data.

X, Y, y = NNF.dataSet(["mnist.csv", 0], 10, True)
X_train, Y_train, y_train, X_test, Y_test, y_test = X[:, :60000], Y[:, :60000], y[:60000], X[:, 60000:], Y[:, 60000:], y[60000:]
print("Accurary on training set:", NN1.simpleEval(X_train, y_train))
print("Accurary on testing set: ", NN1.simpleEval(X_test, y_test))
