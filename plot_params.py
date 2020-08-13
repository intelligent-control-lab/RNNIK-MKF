import matplotlib.pyplot as plt
import numpy as np
import utils as util

plot_wrist_k = 1
plot_elbow_k = 1
plot_error_list = 0
plot_error_dist = 0

if(plot_wrist_k):
    try:
        k_list = np.loadtxt('k_list_wrist.txt')
        x = np.linspace(0, k_list.shape[0], k_list.shape[0])
        fig = plt.figure()
        ax = plt.axes()
        for i in range(0, k_list.shape[1]):
            ax.plot(x, k_list[:, i])
    except Exception as e:
        print('Plot wrist k error!')

if(plot_elbow_k):
    try:
        k_list = np.loadtxt('k_list_elbow.txt')
        x = np.linspace(0, k_list.shape[0], k_list.shape[0])
        fig = plt.figure()
        ax = plt.axes()
        for i in range(0, k_list.shape[1]):
            ax.plot(x, k_list[:, i])
    except Exception as e:
        print('Plot elbow k error!')

if(plot_error_list):
    try:
        et = np.loadtxt('jt_list.txt')
        x = np.linspace(0, et.shape[0], et.shape[0])
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(x, et)
    except Exception as e:
        print('No jt list exists!')
    
if(plot_error_dist):
    try:
        et = np.loadtxt('jt_list.txt')
        fig = plt.figure()
        ax = plt.axes()
        ax.hist(et, bins=int(1//0.01), cumulative=True, density=True)
    except Exception as e:
        print('No jt list exists!')
    
plt.show()