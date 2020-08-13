import numpy as np

class MKF():
    def __init__(self, param_mtx, P, lamda, miu_v, miu_p, sig_r=0.001, sig_q=0.001):
        self.theta = param_mtx # Initial parameter matrix
        self.lamda = lamda # Forgetting factor for RLS
        self.P = P
        self.sig_r = sig_r
        self.sig_q = sig_q
        self.miu_v = miu_v
        self.miu_p = miu_p
        self.first = 1 
        self.V = 0

    def adapt(self, x, err):
        H = np.copy(x)
        tmp = np.linalg.inv(np.matmul(np.matmul(H, self.P), np.transpose(H)) + self.sig_r * np.identity(x.shape[0]))
        K = np.matmul(np.matmul(self.P, np.transpose(H)), tmp) # Kalman gain
        if(self.first):
            self.V = np.matmul(np.transpose(K), np.transpose(err))
            self.first = 0
        else:
            self.V = self.miu_v*self.V + (1-self.miu_v) * np.matmul(np.transpose(K), np.transpose(err))
        
        self.theta = self.theta + self.V
        P = (1/self.lamda) * (self.P - np.matmul(np.matmul(K, H), self.P) + self.sig_q * np.identity(self.P.shape[0]))
        self.P = self.miu_p * self.P + (1-self.miu_p) * P
        return self.theta