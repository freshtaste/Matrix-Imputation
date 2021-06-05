import numpy as np
import pandas as pd
from missing_mat import MissMat
from get_data import get_missing_lr, get_data
from sklearn.decomposition import PCA
from matrix_completion import svt_solve
from tools import SGD


class Impute():
    
    def __init__(self, MissData: MissMat):
        self.MissData = MissData
        
    def fit_via_TW(self, r):
        # PCA on Tall block
        tall = self.MissData.get_tall()
        pca = PCA(n_components=r)
        pca.fit(tall)
        F_tall = pca.transform(tall)
        Lambda_tall = np.linalg.solve(F_tall.T @ F_tall, F_tall.T @ tall).T
        
        # PCA on Wide block
        wide = self.MissData.get_wide().T
        pca.fit(wide)
        Lambda_wide = pca.transform(wide)
        
        # regress Lambda_tall on submatrix of Lambda_wide
        N_o = Lambda_tall.shape[0]
        Lsub = Lambda_wide[:N_o]
        H = np.linalg.solve(Lsub.T @ Lsub, Lsub.T @ Lambda_tall)
        C = F_tall @ H @ Lambda_wide.T
        
        Xout = np.zeros((self.MissData.N, self.MissData.T))
        Xout[self.MissData.W == 1] = self.MissData.X[self.MissData.W == 1]
        Xout[self.MissData.W == 0] = C[self.MissData.W == 0]
        self.Xout = Xout
        self.F = F_tall
        return Xout
    
    def fit_via_TP(self, r):
        # PCA on Tall block
        tall, rest, W_rest = self.MissData.get_tall_rest()
        pca = PCA(n_components=r)
        pca.fit(tall)
        F_tall = pca.transform(tall)
        
        N, T = self.MissData.N, self.MissData.T
        T_o = tall.shape[1]
        Xm = np.zeros((N, T-T_o))
        Ls = np.zeros((T-T_o, r))
        for i in range(T-T_o):
            x, w = rest[:,i], W_rest[:,i]
            f = F_tall[w==1]
            x = x[w==1]
            l = np.linalg.solve(f.T @ f, f.T @ x)
            Ls[i] = l
        Xm = F_tall @ Ls.T
        Xout = np.zeros((self.MissData.N, self.MissData.T))
        Xout[self.MissData.W == 1] = self.MissData.X[self.MissData.W == 1]
        Xout[:,T_o:][self.MissData.W[:,T_o:] == 0] = Xm[self.MissData.W[:,T_o:] == 0]
        self.Xout = Xout
        self.F = F_tall
        return Xout
    
    def fit_via_Weight(self, r):
        N, T = self.MissData.N, self.MissData.T
        cov = pd.DataFrame(self.MissData.X).cov().to_numpy()
        cov[np.isnan(cov)] = np.mean(cov[~np.isnan(cov)])
        eigval, eigvec = np.linalg.eig(cov)
        Lambda = np.sqrt(T)*eigvec[:,:r]
        F = np.zeros((N, r))
        for i in range(N):
            x = np.copy(self.MissData.X[i])
            x[self.MissData.W[i]==0] = 0
            F[i] = np.linalg.solve(Lambda.T @ np.diag(self.MissData.W[i]) @ Lambda,
                    Lambda.T @ x)
        C = F @ Lambda.T
        Xout = np.zeros((self.MissData.N, self.MissData.T))
        Xout[self.MissData.W == 1] = self.MissData.X[self.MissData.W == 1]
        Xout[self.MissData.W == 0] = C[self.MissData.W == 0]
        self.Xout = Xout
        self.F = F
        return Xout
    
    def fit_via_Nuclear(self, r):
        C = svt_solve(self.MissData.X, self.MissData.W)
        Xout = np.zeros((self.MissData.N, self.MissData.T))
        Xout[self.MissData.W == 1] = self.MissData.X[self.MissData.W == 1]
        Xout[self.MissData.W == 0] = C[self.MissData.W == 0]
        self.Xout = Xout
        pca = PCA(n_components=r)
        pca.fit(Xout)
        self.F = pca.transform(Xout)
        return Xout
    
    def fit_via_PQ(self, r):
        F, L = SGD(self.MissData.X_sparse, r, gamma=0.01, lamda=0.1, steps=200)
        C = F @ L
        Xout = np.zeros((self.MissData.N, self.MissData.T))
        Xout[self.MissData.W == 1] = self.MissData.X[self.MissData.W == 1]
        Xout[self.MissData.W == 0] = C[self.MissData.W == 0]
        self.Xout = Xout
        self.F = F
        return Xout
    
    def amputation(self, r):
        row_sum = np.sum(self.MissData.W, axis=0)
        Xout = self.MissData.X[:, row_sum == self.MissData.N]
        self.Xout = Xout
        pca = PCA(n_components=r)
        pca.fit(Xout)
        self.F = pca.transform(Xout)
        return Xout
    
    
if __name__ == "__main__":
    N, T, r = 200, 100, 3
    X = get_data(N, T, r)
    W = get_missing_lr(N, T, r, 150, 80, 0.5)
    Xm = np.copy(X)
    Xm[W==0] = np.nan
    M = MissMat(Xm, W)
    imp = Impute(M)
    Xhat = imp.fit_via_TW(r)
    print(np.mean((Xhat[W==0] - X[W==0])**2))
    Xhat = imp.fit_via_TP(r)
    print(np.mean((Xhat[W==0] - X[W==0])**2))
    Xhat = imp.fit_via_Weight(r)
    print(np.mean((Xhat[W==0] - X[W==0])**2))
    Xhat = imp.fit_via_Nuclear()
    print(np.mean((Xhat[W==0] - X[W==0])**2))
    Xhat = imp.fit_via_PQ(r)
    print(np.mean((Xhat[W==0] - X[W==0])**2))