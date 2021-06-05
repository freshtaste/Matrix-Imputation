import numpy as np


def get_data(N, T, r):
    F = np.random.normal(size=(N, r))
    L = np.random.normal(size=(T, r))
    e = np.random.normal(size=(N, T))
    T = F @ L.T + e
    return T


def get_missing_lr(N, T, r, N_o, T_o, p):
    W = np.ones((N,T))
    Wb = np.random.choice([0,1], size=(N-N_o, T-T_o), p=[p,1-p])
    W[N_o:, T_o:] = Wb
    return W