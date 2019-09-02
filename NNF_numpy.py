import pandas as pd
import numpy as np


def roll(v, layers_size):
    L = len(layers_size)
    LOM = []
    c = 0
    for j in range(L - 1):
        a = int(layers_size[j + 1])
        b = int(layers_size[j])
        matrix = v[c:c + a * b]
        dM = matrix.reshape(a, b)
        LOM.append(dM)
        c = c + a * b
    return LOM


def roll2(v, layers_size):
    L = len(layers_size)
    LOM = []
    c = 0
    for j in range(L - 1):
        a = int(layers_size[j + 1])
        matrix = v[c:c + a]
        dM = matrix.reshape(a, 1)
        LOM.append(dM)
        c = c + a
    return LOM


def unroll(LOM):
    v = np.ones((0, 1))
    for g in range(len(LOM)):
        V = LOM[g].reshape(np.size(LOM[g]), 1)
        v = np.vstack((v, V))
    return v


def f1(init, layers_size):
    if init[0] == "simple":
        f = init[1]
        L = len(layers_size)
        lot_EWM = []
        lot_EBV = []
        for j in range(L - 1):
            a = int(layers_size[j + 1])
            # Set matrix.
            b = int(layers_size[j])
            dM = np.random.rand(a, b) * 2 * f - f
            dV = np.random.rand(a, 1) * 2 * f - f
            # Append to list.
            lot_EWM.append(dM)
            lot_EBV.append(dV)
    return lot_EWM, lot_EBV


def dataSet(file, K):
    oc = file[1]
    data = pd.read_csv(file[0]).values
    y = data[:, oc]
    y = y.reshape(y.shape[0], 1)
    X_a = data[:, range(0, oc)]
    X_b = data[:, range(oc + 1, data.shape[1])]
    X = np.hstack((X_a, X_b)).T
    Y = np.zeros((K, data.shape[0]))
    for h in range(Y.shape[1]):
        index = int(y[h] - 1)
        Y[index, h] = 1
    return X, Y, y


def opt(cf, opt, X, Y, lot_EWM, lot_EBV, layers_size, layers_af):
    import time
    start = time.time()
    if opt[0] == "GD":
        J_history = []
        for iters in range(opt[1]):
            print("Iteration: ", iters + 1)
            J, lot_GOEWM, lot_GOEBV = CF(
                cf,
                lot_EWM,
                lot_EBV, layers_size,
                layers_af, X, Y)
            print("J:", J)
            J_history.append(J)
            v_EWM = unroll(lot_EWM)
            v_EBV = unroll(lot_EBV)
            v_GOEWM = unroll(lot_GOEWM)
            v_GOEBV = unroll(lot_GOEBV)
            v_EWM = v_EWM - opt[2] * v_GOEWM
            v_EBV = v_EBV - opt[2] * v_GOEBV
            lot_EWM = roll(v_EWM, layers_size)
            lot_EBV = roll2(v_EBV, layers_size)
    end = time.time()
    print("The total time it took:", end - start)
    return lot_EWM, lot_EBV, J_history


def CF(cf, lot_EWM, lot_EBV, layers_size, layers_af, X, Y):
    if cf == "CE":
        L = len(layers_size)
        listZ = [0] * L
        listA = [0] * L
        listDZ = [0] * L
        lot_GOEWM = [0] * len(lot_EWM)
        lot_GOEBV = [0] * len(lot_EBV)
        m = X.shape[1]
        # FP.
        listZ[0] = X
        listA[0] = af(layers_af, listZ[0], 0)
        for j in range(L - 1):
            listZ[j + 1] = lot_EWM[j] @ listA[j] + lot_EBV[j]
            listA[j + 1] = af(layers_af, listZ[j + 1], j + 1)
        J = (1 / m) * sum(sum(-Y * np.log(listA[L - 1]) - (1 - Y) * np.log((1 - listA[L - 1]))))
        # BP.
        listDZ[L - 1] = listA[L - 1] - Y
        lot_GOEWM[L - 2] = (1 / m) * (listDZ[L - 1] @ listA[L - 2].T)
        lot_GOEBV[L - 2] = (1 / m) * np.sum(listDZ[L - 1], axis=1, keepdims=True)
        for j in reversed(range(1, L - 1)):
            listDZ[j] = (lot_EWM[j].T @ listDZ[j + 1]) * (listA[j] * (1 - listA[j]))
            lot_GOEWM[j - 1] = (1 / m) * (listDZ[j] @ listA[j - 1].T)
            lot_GOEBV[j - 1] = (1 / m) * np.sum(listDZ[j], axis=1, keepdims=True)
    return J, lot_GOEWM, lot_EBV


def af(layers_af, tensor, nthNumber):
    if layers_af[nthNumber] == "identity":
        tensorModified = tensor
    if layers_af[nthNumber] == "sigmoid":
        tensorModified = 1 / (1 + np.exp(-tensor))
    return tensorModified
