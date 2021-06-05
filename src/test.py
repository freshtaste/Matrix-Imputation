import numpy as np
from imputation import Impute
from get_data import get_data, get_missing_lr
from missing_mat import MissMat


def model1(N, T, N_o, T_o, r, n=10):
    MSE = np.zeros((n, 5))
    for i in range(n):
        X = get_data(N, T, r)
        W = get_missing_lr(N, T, r, N_o, T_o, 0.5)
        Xm = np.copy(X)
        Xm[W==0] = np.nan
        M = MissMat(Xm, W)
        imp = Impute(M)
        Xhat0 = imp.fit_via_TW(r)
        MSE[i, 0] = np.mean((Xhat0[W==0] - X[W==0])**2)
        Xhat1 = imp.fit_via_TP(r)
        MSE[i, 1] = np.mean((Xhat1[W==0] - X[W==0])**2)
        Xhat2 = imp.fit_via_Weight(r)
        MSE[i, 2] = np.mean((Xhat2[W==0] - X[W==0])**2)
        Xhat3 = imp.fit_via_Nuclear()
        MSE[i, 3] = np.mean((Xhat3[W==0] - X[W==0])**2)
        Xhat4 = imp.fit_via_PQ(r)
        MSE[i, 4] = np.mean((Xhat4[W==0] - X[W==0])**2)
    return MSE

# increase p
def model2(p, r, n=10):
    MSE = np.zeros((n, 5))
    for i in range(n):
        X = get_data(200, 100, r)
        W = get_missing_lr(200, 100, r, 100, 50, p)
        Xm = np.copy(X)
        Xm[W==0] = np.nan
        M = MissMat(Xm, W)
        imp = Impute(M)
        Xhat0 = imp.fit_via_TW(r)
        MSE[i, 0] = np.mean((Xhat0[W==0] - X[W==0])**2)
        Xhat1 = imp.fit_via_TP(r)
        MSE[i, 1] = np.mean((Xhat1[W==0] - X[W==0])**2)
        Xhat2 = imp.fit_via_Weight(r)
        MSE[i, 2] = np.mean((Xhat2[W==0] - X[W==0])**2)
        Xhat3 = imp.fit_via_Nuclear()
        MSE[i, 3] = np.mean((Xhat3[W==0] - X[W==0])**2)
        Xhat4 = imp.fit_via_PQ(r)
        MSE[i, 4] = np.mean((Xhat4[W==0] - X[W==0])**2)
    return MSE


def model3(N, T, T_o, r, n=10):
    MSE = np.zeros((n, 4))
    for i in range(n):
        X = get_data(N, T, r)
        W = get_missing_lr(N, T, r, 0, T_o, 0.5)
        Xm = np.copy(X)
        Xm[W==0] = np.nan
        M = MissMat(Xm, W)
        imp = Impute(M)
        Xhat1 = imp.fit_via_TP(r)
        MSE[i, 0] = np.mean((Xhat1[W==0] - X[W==0])**2)
        Xhat2 = imp.fit_via_Weight(r)
        MSE[i, 1] = np.mean((Xhat2[W==0] - X[W==0])**2)
        Xhat3 = imp.fit_via_Nuclear()
        MSE[i, 2] = np.mean((Xhat3[W==0] - X[W==0])**2)
        Xhat4 = imp.fit_via_PQ(r)
        MSE[i, 3] = np.mean((Xhat4[W==0] - X[W==0])**2)
    return MSE


def model4(N, T, r, p=0.5, n=10):
    MSE = np.zeros((n, 3))
    for i in range(n):
        X = get_data(N, T, r)
        W = get_missing_lr(N, T, r, 0, 0, p)
        Xm = np.copy(X)
        Xm[W==0] = np.nan
        M = MissMat(Xm, W)
        imp = Impute(M)
        Xhat2 = imp.fit_via_Weight(r)
        MSE[i, 0] = np.mean((Xhat2[W==0] - X[W==0])**2)
        Xhat3 = imp.fit_via_Nuclear()
        MSE[i, 1] = np.mean((Xhat3[W==0] - X[W==0])**2)
        Xhat4 = imp.fit_via_PQ(r)
        MSE[i, 2] = np.mean((Xhat4[W==0] - X[W==0])**2)
    return MSE


        
if __name__ == "__main__":
    """
    print("Model1: ")
    MSE = np.zeros((5,5))
    MSE[0] = np.mean(model1(200, 100, 20, 10, 3),axis=0)
    MSE[1] = np.mean(model1(200, 100, 50, 25, 3),axis=0)
    MSE[2] = np.mean(model1(200, 100, 100, 50, 3),axis=0)
    MSE[3] = np.mean(model1(200, 100, 100, 75, 3),axis=0)
    MSE[4] = np.mean(model1(200, 100, 150, 75, 3),axis=0)
    print(MSE)
    
    
    print("Model2: ")
    MSE = np.zeros((5,5))
    MSE[0] = np.mean(model2(0.1, 3),axis=0)
    MSE[1] = np.mean(model2(0.3, 3),axis=0)
    MSE[2] = np.mean(model2(0.5, 3),axis=0)
    MSE[3] = np.mean(model2(0.7, 3),axis=0)
    MSE[4] = np.mean(model2(0.9, 3),axis=0)
    print(MSE)
    
    
    
    print("Model1.5: ")
    MSE = np.zeros((5,5))
    MSE[0] = np.mean(model1(200, 100, 100, 50, 3),axis=0)
    MSE[1] = np.mean(model1(200, 100, 100, 50, 5),axis=0)
    MSE[2] = np.mean(model1(200, 100, 100, 50, 10),axis=0)
    MSE[3] = np.mean(model1(200, 100, 100, 50, 20),axis=0)
    MSE[4] = np.mean(model1(200, 100, 100, 50, 30),axis=0)
    print(MSE)
    
    
    
    print("Model3: ")
    MSE3 = np.zeros((5,4))
    MSE3[0] = np.mean(model3(200, 100, 10, 3),axis=0)
    MSE3[1] = np.mean(model3(200, 100, 25, 3),axis=0)
    MSE3[2] = np.mean(model3(200, 100, 50, 3),axis=0)
    MSE3[3] = np.mean(model3(200, 100, 75, 3),axis=0)
    MSE3[4] = np.mean(model3(200, 100, 90, 3),axis=0)
    print(MSE3)
    
    
    
    print("Model4: ")
    MSE4 = np.zeros((6,3))
    #MSE4[0] = np.mean(model4(50, 20, 3),axis=0)
    #MSE4[1] = np.mean(model4(100, 50, 3),axis=0)
    #MSE4[2] = np.mean(model4(200, 100, 3),axis=0)
    MSE4[3] = np.mean(model4(200, 100, 3, p=0.9),axis=0)
    MSE4[4] = np.mean(model4(200, 100, 3, p=0.8),axis=0)
    MSE4[5] = np.mean(model4(200, 100, 3, p=0.7),axis=0)
    print(MSE4)
    
    """
    