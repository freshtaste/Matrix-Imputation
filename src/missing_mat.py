import numpy as np
from scipy.sparse import coo_matrix


class MissMat():
    
    def __init__(self, X, W):
        self.X = X
        self.W = W
        self.N, self.T = X.shape
        self.verify_data()
        self.X_sparse = self.get_sparse()
        
    
    def verify_data(self):
        if type(self.X) is not np.ndarray or type(self.W) is not np.ndarray:
            raise RuntimeError("Data is not numpy array, type: {}, {}."
                               .format(type(self.X), type(self.W)))
            
        if self.X.shape != self.W.shape:
            raise RuntimeError("Shape not equal, {}, {}".
                               format(self.X.shape, self.W.shape))
        
        if set(self.W.reshape(-1,1).squeeze()) != {0,1}:
            raise RuntimeError("Value of W is not binary, {}.".format(set(self.W.reshape(-1,1).squeeze())))
            
    
    def sort(self):
        # sort columns
        row_sum = np.sum(self.W, axis=0)
        self.idx_col = np.argsort(-row_sum)
        W_sort_col = self.W[:,self.idx_col]
        X_sort_col = self.X[:, self.idx_col]
        # sort rows
        col_sum = np.sum(W_sort_col, axis=1)
        self.idx_row = np.argsort(-col_sum)
        W_sorted = W_sort_col[self.idx_row, :]
        X_sorted = X_sort_col[self.idx_row, :]
        self.W = W_sorted
        self.X = X_sorted
        
    
    def un_sort(self, X, W):
        X_tmp = X[self.idx_row,:]
        W_tmp = W[self.idx_row,:]
        X_unsort = X_tmp[:,self.idx_col]
        W_unsort = W_tmp[:,self.idx_col]
        return X_unsort, W_unsort
    
        
    def get_sparse(self):
        M = np.copy(self.X)
        M[self.W==0] = 0
        return coo_matrix(M)
        
        
    def demean(self):
        mu = np.mean(self.X[self.W==1])
        self.X -= mu
        
        
    def get_tall(self):
        row_sum = np.sum(self.W, axis=0)
        return self.X[:,row_sum==self.N]
    
    
    def get_wide(self):
        col_sum = np.sum(self.W, axis=1)
        return self.X[col_sum==self.T, :]
    
    
    def get_tall_rest(self):
        row_sum = np.sum(self.W, axis=0)
        return self.X[:,row_sum==self.N], self.X[:,row_sum<self.N], self.W[:,row_sum<self.N]
    
    

if __name__ == "__main__":
    X = np.arange(25).reshape(5,5)
    W = np.array([[0, 0, 0, 0, 0],
       [0, 1, 1, 1, 0],
       [1, 0, 1, 1, 0],
       [1, 0, 1, 0, 0],
       [1, 0, 1, 0, 0]])
    M = MissMat(X, W)
    M.sort()
    print(X)
    print(W)
    print(M.X)
    print(M.W)
    print(M.get_tall())
    print(M.get_wide())
    print(M.X_sparse)
    print(M.un_sort(M.X, M.W))