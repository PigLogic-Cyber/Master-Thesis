import os
import time
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns


# data2 = np.loadtxt('train.txt', delimiter=' ', dtype='str')
# [row2, col2] = data2.shape
# print('row2 {},col2 {}'.format(row2, col2))
# angle2 = data2[1:row2, 1].astype('float')
# plt.hist(angle2, bins=np.arange(min(angle2), max(angle2), 1))   #need change something
# plt.title("train angle Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
#
# plt.show()


data2 = np.loadtxt('val (copy).txt', delimiter=' ', dtype='str')
[row2, col2] = data2.shape
print('row2 {},col2 {}'.format(row2, col2))
angle2 = data2[0:row2, 1].astype('float')
print(angle2)
