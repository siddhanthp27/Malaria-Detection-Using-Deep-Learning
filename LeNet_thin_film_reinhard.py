"""
File for training LeNet Classifier on Reinhard Stain Normalized Patches.
"""

# coding: utf-8

# In[1]:


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
from random import shuffle


# In[2]:


class LeNet(nn.Module):
    
    def __init__(self):
        
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        n_size = self.get_shape((1, 80, 80))
        
        self.fc1 = nn.Linear(n_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

        #torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        #torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        #torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        #torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        
    def get_shape(self, input_shape):
        
        bs = 1
        input_tensor = Variable(torch.rand(bs, *input_shape))
        output_feat = self.forward_features(input_tensor)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
    
    def forward_features(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
#         print(out.shape)
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


# In[3]:


class Dataset(Dataset):
    
    def __init__(self, transform=None):
        self.folders = ['malaria/', 'non-malaria/']
        self.inverse_transform = {'non-malaria': 0, 'malaria': 1}
        self.root = 'patches/'
        self.files = []
        for folder in self.folders:
            temp = os.listdir(self.root + folder)
            for i in range(len(temp)):
#                 if '.png' in temp[i]:
                img_file = io.imread(self.root + folder + temp[i])
                # print(img_file.shape)
                # # Filter to check for predefined patch  size
                if img_file.shape == (80, 80):
                    self.files.append(self.root + folder + temp[i])
        #temp_non_malaria = os.listdir(self.root + 'non-malaria/')
        #shuffle(temp_non_malaria)
        #temp_non_malaria = temp_non_malaria[:len(self.files)]
        #for i in range(len(temp_non_malaria)):
        #    self.files.append(self.root + folder + temp[i])
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_file = io.imread(img_name)
        
#         if img_file is not None:
#             img_file = np.array(img_file, dtype=np.float32)
# #             img_file = np.expand_dims(img_file, axis=2)
#             img /= 255
            
        #mean = np.mean(img_file)
        #img_file = img_file - mean	
		# # Expand dims for grayscale image
        img_file = np.expand_dims(img_file, axis=2)
        if self.transform:
            img_file = self.transform(img_file)
                
        sample = {'image': img_file, 'label': self.inverse_transform[img_name.split('/')[1]]}
            
        return sample                                                                 


# In[4]:

# Change type to tensor
trans_1 = transforms.ToTensor()
trans_2 = transforms.RandomCrop((111, 111), padding=(0, 0), pad_if_needed=True)
trans = transforms.Compose([trans_1])

transformed_dataset = Dataset(trans)


# In[65]:


dataloader = DataLoader(transformed_dataset, batch_size=16, shuffle=True)


# In[70]:


model = LeNet()
model.cuda()


# In[67]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# In[68]:


def train(epoch):
    model.train()
    for i, sample in enumerate(dataloader):
        image = sample['image']
        label = sample['label']
        
        image, label = Variable(image).cuda(), Variable(label).cuda()
        
        optimizer.zero_grad()
        output = model(image)
        
        loss = criterion(output, label)
        
        if i % 100 == 0:
            print('Train - Epoch: %d, Batch: %d, Loss: %f' % (epoch, i, loss.data[0]))
            
        loss.backward()
        optimizer.step()


# In[69]:


for e in range(50):
    train(e)


# In[ ]:


torch.save(model, 'thin_film_LeNet.pt')


# In[ ]:


torch.save(model.state_dict(), 'thin_film_LeNet_dict.pt')

