"""File for converting pkl files into images using matplotlib"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# file_list = ['mdb001', 'mdb002', 'mdb005', 'mdb010', 'mdb011', 'mdb012']
# file_list = ['8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png', '6b14c855-8561-417c-97a4-63fa552842fd.png', '2559636b-f01a-4414-93da-210c3b12d153.png']
# file_list = ['8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png', '6b14c855-8561-417c-97a4-63fa552842fd.png']

file_list = os.listdir('macenko_pkl_dump')

for i in file_list:
    file_list[i] = 'macenko_pkl_dump' + file_list[i]

for file_name in file_list:
    class_map = pickle.load(open(file_name + "_classmap.pkl", 'rb'))

    #area = np.zeros((750, 750))

    #for i in range(750):
    #    for j in range(750):
    #        if (class_map[i][j] == 1.0):
    #            area[i][j] = 1

    from pylab import rcParams
    rcParams['figure.figsize'] = 1200./96, 1600./96

    plt.axis('equal')

    fig, ax = plt.subplots()

    im = ax.imshow(class_map, cmap='jet')

    ax.set(adjustable='box-forced', aspect='equal')

    fig.colorbar(im, ax=ax)

    plt.axis('equal')
    plt.tight_layout()

    plt.savefig('./Lenet_classmap_{}.png'.format(file_name), dpi=96)
