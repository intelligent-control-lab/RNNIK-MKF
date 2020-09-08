import matplotlib.pyplot as plt
import numpy as np

plot_wrist_k = 1
plot_elbow_k = 1
plot_wrist_adapt = 1
plot_elbow_adapt = 1

if(plot_wrist_k):
    try:
        k_list = np.loadtxt('k_list_wrist.txt')
        x = np.linspace(0, k_list.shape[0], k_list.shape[0])
        fig = plt.figure()
        fig.suptitle('Wrist Parameters')
        ax = plt.axes()
        for i in range(0, k_list.shape[1]):
            ax.plot(k_list[:, i])
    except Exception as e:
        print('Plot wrist parameters failed!')

if(plot_elbow_k):
    try:
        k_list = np.loadtxt('k_list_elbow.txt')
        x = np.linspace(0, k_list.shape[0], k_list.shape[0])
        fig = plt.figure()
        fig.suptitle('Arm Parameters')
        ax = plt.axes()
        for i in range(0, k_list.shape[1]):
            ax.plot(k_list[:, i])
    except Exception as e:
        print('Plot arm parameters failed!')

if(plot_wrist_adapt):
    try:
        wrist_adapt_err = np.loadtxt('wrist_adapt_err.txt')
        wrist_noadapt_err = np.loadtxt('wrist_noadapt_err.txt')
        wrist_adapt_error = np.zeros((wrist_adapt_err.shape[0], 1))
        wrist_noadapt_error = np.zeros((wrist_adapt_err.shape[0], 1))
        for i in range(wrist_adapt_err.shape[0]):
            try:
                wrist_adapt_error[i] = np.sqrt((wrist_adapt_err[i, 0]**2 + wrist_adapt_err[i, 1]**2 + wrist_adapt_err[i, 2]**2)/3)
                wrist_noadapt_error[i] = np.sqrt((wrist_noadapt_err[i, 0]**2 + wrist_noadapt_err[i, 1]**2 + wrist_noadapt_err[i, 2]**2)/3)
            except:
                pass
        fig = plt.figure()
        fig.suptitle('Wrist Adaptation Error')
        ax = plt.axes()
        ax.plot(wrist_adapt_error, label='MKF')
        ax.plot(wrist_noadapt_error, label='Unadapt')
        ax.legend(loc="upper right", shadow=True, fancybox=True)
        ax.set(ylabel='Wrist Adaptation RMSE')
    except Exception as e:
        print('Plot wrist adaptation error failed!')
    
if(plot_elbow_adapt):
    try:
        elbow_adapt_err = np.loadtxt('elbow_adapt_err.txt')
        elbow_noadapt_err = np.loadtxt('elbow_noadapt_err.txt')
        elbow_adapt_error = np.zeros((elbow_adapt_err.shape[0], 1))
        elbow_noadapt_error = np.zeros((elbow_adapt_err.shape[0], 1))
        for i in range(elbow_adapt_err.shape[0]):
            try:
                elbow_adapt_error[i] = np.sqrt((elbow_adapt_err[i, 0]**2 + elbow_adapt_err[i, 1]**2 + elbow_adapt_err[i, 2]**2 + elbow_adapt_err[i, 3]**2 + elbow_adapt_err[i, 4]**2)/5)
                elbow_noadapt_error[i] = np.sqrt((elbow_noadapt_err[i, 0]**2 + elbow_noadapt_err[i, 1]**2 + elbow_noadapt_err[i, 2]**2 + elbow_noadapt_err[i, 3]**2 + elbow_noadapt_err[i, 4]**2)/5)
            except:
                pass
        fig = plt.figure()
        fig.suptitle('Arm Adaptation Error')
        ax = plt.axes()
        ax.plot(elbow_adapt_error, label='MKF')
        ax.plot(elbow_noadapt_error, label='Unadapt')
        ax.legend(loc="upper right", shadow=True, fancybox=True)
        ax.set(ylabel='Arm Adaptation RMSE (rad)')
    except Exception as e:
        print("Plot arm adaptation error failed!")
plt.show()