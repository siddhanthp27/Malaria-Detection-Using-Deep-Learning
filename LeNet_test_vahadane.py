""" 
File for testing trained LeNet (using Vahadane Stain Normalization) on images by iterating over every pixel
"""

import torch
import torch.nn as nn
import os
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
import torch.optim as optim
import cv2
import scipy.misc
from torch.utils.data import Dataset
from skimage import io, transform
import pickle
import time
from skimage.measure import label, regionprops
from staintool.brightness_standardizer import BrightnessStandardizer
from staintool.reinhard_color_normalizer import ReinhardColorNormalizer

stain_normalizer = ReinhardColorNormalizer()

standardizer = BrightnessStandardizer()

color_dict = dict()

color_dict[0.0] = np.array([0, 0, 0])

# Load BGR values for red color
color_dict[1.0] = np.array([0, 0, 255])

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        n_size = self.get_shape((1, 80, 80))
        print(n_size)

        self.fc1 = nn.Linear(n_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def get_shape(self, input_shape):
        bs = 1
        input_tensor = Variable(torch.rand(bs, *input_shape))
        output_feat = self.forward_feature(input_tensor)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward_feature(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        return out

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out))
        return out


model = LeNet()
model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# load pretrained model
model.load_state_dict(torch.load('vahadane_dump/thin_film_LeNet_dict.pt'))

trans = transforms.ToTensor()

# file_list = ['8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png', '6b14c855-8561-417c-97a4-63fa552842fd.png']

#file_list = ['2559636b-f01a-4414-93da-210c3b12d153.png', '0a747cb3-c720-4572-a661-ab5670a5c42e.png', '0d2aba33-6920-4001-bd54-59fe0bf9f50e.png']

file_list = os.listdir('vahadane/')

# file_list = file_list[:100]

start = time.time()

done_list = os.listdir('vahadane_pkl_dump/')

for file_name in file_list:

#	Check if files already present in the folder
    fin_name = file_name + '_classmap.pkl'
 
    if fin_name in done_list:
        continue

    if file_name.split('.')[-1] != 'png' and file_name.split('.')[-1] != 'jpg':
        continue

    img = cv2.imread('vahadane/' + file_name)
    if img is None:
        continue

    orig_img = img
# Convert file to gray
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if orig_img.shape != (1200, 1600, 3):
        continue
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #lab_planes = cv2.split(lab)

    #clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(45, 45))

    #lab_planes[0] = clahe.apply(lab_planes[0])

    #lab = cv2.merge(lab_planes)

    #img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    class_map = np.zeros((1200, 1600))
    heat_map = np.zeros((1200, 1600)) 

    for i in range(40, 1200-40):
        for j in range(40, 1600-40):

# 			Create 40x40 patch around every pixel
            patch = img[i-40:i+40, j-40:j+40]

            # patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

            # mean = np.mean(patch)

            # patch = patch - mean

            patch = np.expand_dims(patch, axis=2)

            patch = trans(patch)

            patch = patch.unsqueeze(0)
            # patch = patch.float()
            # patch = patch.unsqueeze(0)

            patch = Variable(patch).cuda()

            output = model(patch)

            np_label = output.cpu().detach().numpy()

            heat_map[i][j] = np_label[0][1]

            # print(heat_map[i][j])

            c = np.argmax(np_label)

            class_map[i][j] = c

            # print(class_map[i][j])

    #overlay = np.zeros((1200, 1600, 3))

    #for i in range(class_map.shape[0]):
    #    for j in range(class_map.shape[1]):
    #        if (class_map[i][j] == 1.0):
    #            overlay[i][j] = color_dict[1.0]

#heatmap_img = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)

    #weighted = cv2.addWeighted(orig_img, 0.7, overlay, 0.3, 0, dtype=cv2.CV_64F)

    #cv2.imwrite('vahadane_dump/'+file_name + '_overlay.png', weighted)

    #prop_map = np.zeros((1200, 1600))

    # #   Generate region props

    #lbl = label(class_map, connectivity=2)

    #props = regionprops(lbl)

# #   Apply Area Filter on the prop map

    #for i in range(len(props)):
    #    for j in range(len(props[i].coords)):
    #        if (props[i].area >= 4900):
    #            prop_map[props[i].coords[j][0]][props[i].coords[j][1]] = 1

#    overlay = np.zeros((1200, 1600, 3))

# # Generate equivalent RGB matrix for the prop map

#    for i in range(class_map.shape[0]):
#        for j in range(class_map.shape[1]):
#            if (prop_map[i][j] == 1.0):
#                overlay[i][j] = color_dict[1.0]

#heatmap_img = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)

# # Overlay the image with the RGB Prop matrix

#    weighted = cv2.addWeighted(orig_img, 0.7, overlay, 0.3, 0, dtype=cv2.CV_64F)

#    cv2.imwrite('vahadane_dump/'+file_name + '_overlay_area_filter.png', weighted)

    
    f = open('vahadane_pkl_dump/'+file_name + '_classmap.pkl', 'wb')
    pickle.dump(class_map, f)
    f.close()

    f = open('vahadane_pkl_dump/'+file_name + '_heatmap.pkl', 'wb')
    pickle.dump(heat_map, f)
    f.close()

end = time.time()

print('Time elapsed = ' + str(end-start))
