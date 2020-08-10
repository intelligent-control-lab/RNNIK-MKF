import matplotlib.pyplot as plt
import numpy as np

k_list = np.loadtxt('k_list_wrist.txt')
x = np.linspace(0, k_list.shape[0], k_list.shape[0])
fig = plt.figure()
ax = plt.axes()
for i in range(0, k_list.shape[1]):
    ax.plot(x, k_list[:, i])

k_list = np.loadtxt('k_list_elbow.txt')
x = np.linspace(0, k_list.shape[0], k_list.shape[0])
fig = plt.figure()
ax = plt.axes()
for i in range(0, k_list.shape[1]):
    ax.plot(x, k_list[:, i])
plt.show()
