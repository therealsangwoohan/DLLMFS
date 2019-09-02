import torch
import pandas as pd
from tqdm import tqdm
import time as time
import pickle


# data = [filename, oc]
def dataSet(data, K, normalize=False):
    print("Processing the data ...")
    dataTensor = torch.tensor(pd.read_csv(data[0]).values)
    if data[1] != "None":
        oc = data[1]
        X_a = dataTensor[:, range(0, oc)]
        X_b = dataTensor[:, range(oc + 1, dataTensor.shape[1])]
        X = torch.cat((X_a, X_b), 1).t().float()
        if normalize == "True":
            print("Normalizing input data ...")
            X = (X - X.mean(0)) / X.std(0)
        y = dataTensor[:, oc]
        y = y.reshape(y.shape[0], 1)
        yt = y.t()
        yt = yt.repeat(K, 1)
        y_c = torch.arange(K)
        y_c = y_c.reshape(len(y_c), 1)
        y_c = y_c.repeat((1, y.shape[0]))
        Y = torch.eq(yt, y_c)
        print("Finished processing the data!")
        return X.cuda(), Y.float().cuda(), y.float().cuda()
    else:
        X = dataTensor.t()
        print("Finished processing the data!")
        return X.float().cuda()


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
    v = torch.ones((0, 1)).cuda()
    for g in range(len(LOM)):
        V = LOM[g].reshape(LOM[g].numel(), 1)
        v = torch.cat((v, V), 0)
    return v


def f1(init, layers_size):
    # start = time.time()
    if init[0] == 1:
        f = init[1]
        L = len(layers_size)
        lot_EWM = []
        lot_EBV = []
        for j in range(L - 1):
            a = int(layers_size[j + 1])
            # Set matrix.
            b = int(layers_size[j])
            dM = (torch.rand(a, b) * 2 * f - f).cuda()
            dV = (torch.rand(a, 1) * 2 * f - f).cuda()
            # Append to list.
            lot_EWM.append(dM)
            lot_EBV.append(dV)
    # end = time.time()
    # print("The total time it took:", end - start)
    return lot_EWM, lot_EBV


def opt(cf, opt, X, Y, lot_EWM, lot_EBV, layers_size, layers_af, lam, p):
    start = time.time()
    if opt[0] == 1:
        J_history = []
        for iters in tqdm(range(int(opt[1]))):
            J, lot_GOEWM, lot_GOEBV = CF(
                cf,
                lot_EWM,
                lot_EBV,
                layers_size,
                layers_af,
                X,
                Y,
                lam,
                p)
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
    print("The total time it took for training:", end - start)
    return lot_EWM, lot_EBV, J_history


def CF(cf, lot_EWM, lot_EBV, layers_size, layers_af, X, Y, lam, p):
    if cf == 1:
        L = len(layers_size)
        m = X.shape[1]
        # listZ, listA, listDA, listDZ = [0] * L, [0] * L, [0] * L, [0] * L
        listZ, listA, listDZ = [0] * L, [0] * L, [0] * L
        lot_GOEWM, lot_GOEBV = [0] * (L - 1), [0] * (L - 1)
        # FP.
        listZ[0] = X
        listA[0] = af(layers_af, listZ[0], 0)
        # if p != 1:
        #     listDA[0] = (torch.rand(listA[0].size()).cuda() < p).float()
        #     listA[0] = (listA[0] * listDA[0]) / p
        for j in range(L - 1):
            listZ[j + 1] = lot_EWM[j] @ listA[j] + lot_EBV[j]
            listA[j + 1] = af(layers_af, listZ[j + 1], j + 1)
            # if p != 1:
            #     listDA[j + 1] = (torch.rand(listA[j + 1].size()).cuda() < p).float()
            #     listA[j + 1] = (listA[j + 1] * listDA[j + 1]) / p
        # BP.
        listDZ[L - 1] = listA[L - 1] - Y
        lot_GOEWM[L - 2] = (1 / m) * (listDZ[L - 1] @ listA[L - 2].t()) \
            + (lam / m) * lot_EWM[L - 2]
        lot_GOEBV[L - 2] = (1 / m) * torch.sum(listDZ[L - 1], 1, True)
        for j in reversed(range(1, L - 1)):
            listDZ[j] = (lot_EWM[j].t() @ listDZ[j + 1]) \
                * (listA[j] * (1 - listA[j]))
            lot_GOEWM[j - 1] = (1 / m) * (listDZ[j] @ listA[j - 1].t()) \
                + (lam / m) * lot_EWM[j - 1]
            lot_GOEBV[j - 1] = (1 / m) * torch.sum(listDZ[j], 1, True)
        # J = (1 / m) * sum(sum(-Y * torch.log(listA[L - 1])
        # - (1 - Y) * torch.log((1 - listA[L - 1]))))
        J = "Not computed"
    return J, lot_GOEWM, lot_EBV


def af(layers_af, tensor, nthNumber):
    if layers_af[nthNumber] == 1:
        tensorModified = tensor
    if layers_af[nthNumber] == 2:
        tensorModified = 1 / (1 + torch.exp(-tensor))
    return tensorModified


def load(fileName):
    pi = open(fileName, "rb")
    return pickle.load(pi)
