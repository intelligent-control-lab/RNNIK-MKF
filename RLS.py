import numpy as np

class RLS():
    def __init__(self, param_mtx, F, lamda):
        self.theta = param_mtx
        self.lamda = lamda # Forgetting factor for RLS
        self.F = F # Learning gain for RLS
    
    def adapt(self, x, err):
        upper = np.matmul(np.matmul(np.matmul(self.F, x), np.transpose(x)), self.F)
        bottom = self.lamda + np.matmul(np.matmul(np.transpose(x), self.F), x)
        self.F = 1/self.lamda*(self.F-upper/bottom)
        self.theta = self.theta+np.matmul(np.matmul(self.F, x), np.transpose(err))
        return self.theta