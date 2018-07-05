import numpy as np
import matplotlib.pyplot as plt
import sys
import caffe
from PIL import Image
import cv2
import statistics
import time
import math


drop_thresh = 0.1   #(0~1)


compression_net = caffe.Net('pilotnet_deploy_s_drop01.prototxt','snapshot/s_dropout01_20000.caffemodel',caffe.TEST)
nocompression_net = caffe.Net('pilotnet_deploy_s.prototxt','snapshot/sully_cropdata_20000.caffemodel',caffe.TEST)
fc6 = nocompression_net.params['fc6'][0].data[...]





print('one matrix shape is {}'.format(fc6.shape))
fc7 = nocompression_net.params['fc7'][0].data[...]
print('another one matrix shape is {}'.format(fc7.shape))
kkk = np.dot(fc7,fc6)
print('kkk matrix shape is {}'.format(kkk.shape))
u, s, vh = np.linalg.svd(kkk)
print('u shape is {}'.format(u.shape))
print('s shape is {}'.format(s.shape))
print('vh shape is {}'.format(vh.shape))

[m,n] = fc6.shape
print('traget shape is {}'.format(fc6.shape))
add1_x = int(m*drop_thresh)

kk = np.zeros((add1_x,vh.shape[0]))
kk[0:add1_x,0:add1_x] = np.diag(s)[0:add1_x,0:add1_x]

a = u[:,0:add1_x]
b = np.dot(kk,vh)

print(kk.shape)
print(a.shape)
print(b.shape)

params = []
for k,v in nocompression_net.params.items():
    params.append(k)
    print(k,v[0].data.shape,v[1].data.shape)

for pr in params:
    if pr != 'fc6' and pr != 'fc7':
        compression_net.params[pr] = nocompression_net.params[pr]
        print('{} layer has been transfered'.format(pr))

bios_matrix = nocompression_net.params['fc6'][1].data[...]
bios_matrix = bios_matrix[0:add1_x]

fix1 = np.zeros(add1_x)
fix2 = np.zeros(len(nocompression_net.params['fc7'][1].data[...]))

compression_net.params['fc6'][0].data[...] = b
# compression_net.params['fc6'][1].data[...] = bios_matrix
compression_net.params['fc6'][1].data[...] = fix1
compression_net.params['fc7'][0].data[...] = a
# compression_net.params['fc7'][1].data[...] = nocompression_net.params['fc7'][1].data[...]
compression_net.params['fc7'][1].data[...] = fix2
compression_net.save('new_model/myDropoutcompression.caffemodel')




# big_index = []
# big_item = []
# small_index = []
# small_item = []
# [m,n] = fc6.shape
# print('traget shape is {}'.format(fc6.shape))
# start = time.time()
# for index, item in enumerate(fc6.reshape((m*n,1))):
#     # print(index)
#     if item > drop_thresh*:
#         big_index.append(index)
#         big_item.append(item)
#     else:
#         small_index.append(index)
#         small_item.append(item)
# print('big item number is {}'.format(len(big_item)))
# print('small item number is {}'.format(len(small_item)))
# end = time.time()
# t = end -start
# print('execution time: %0.3f'%(t))
