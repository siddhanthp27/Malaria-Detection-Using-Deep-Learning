#!/usr/bin/env python
# coding: utf-8

# This file calculates the number of false positives, false negatives and true positives for a given image.

# In[3]:


import pickle
from skimage.measure import label, regionprops
import os
import csv
import numpy as np

bb_dict = pickle.load(open('image_bb_dict.pkl'))

# In[6]:

# Get file names
file_list = os.listdir('vahadane_pkl_dump')
root_dir = 'vahadane_pkl_dump/'

mycsv = csv.writer(open('auroc_analysis/counter_vahadane_areathresh=4900.csv', 'wb'))
mycsv.writerow(['Image Name', 'True Positive', 'False Positive', 'False Negative', 'Total'])

thresh = 4900

for file_name in file_list:

    if file_name.split('_')[1] != 'classmap.pkl':
        continue

    # Get key from file names

    key = file_name.split('_')[0]

    # In[14]:


    # Get list of ground truth bounding boxes: (minc, minr, maxc, maxr)

    bb_coords = bb_dict[key]

    bb_coords_list = []

    for i in range(len(bb_coords)):
        if (bb_coords[i]['category'] == 'schizont') or (bb_coords[i]['category'] == 'difficult') or (bb_coords[i]['category'] == 'ring') or     (bb_coords[i]['category'] == 'gametocyte') or (bb_coords[i]['category'] == 'trophozoite'):
            bb_coords_list.append((bb_coords[i]['bounding_box']['minimum']['c'], bb_coords[i]['bounding_box']['minimum']['r'],                              bb_coords[i]['bounding_box']['maximum']['c'], bb_coords[i]['bounding_box']['maximum']['r']))

    # bb_coords_list


    # In[19]:


    # Load the class map

    class_map = pickle.load(open(root_dir + file_name, 'rb'))


    # In[20]:


    # Calculate the region proposals
    lbl = label(class_map, connectivity=2)

    props = regionprops(lbl)

    prop_map = np.zeros((1200, 1600))

    #for i in range(len(props)):
    #    for j in range(len(props[i].coords)):
    #        if (props[i].area >= 100):
    #            prop_map[props[i].coords[j][0]][props[i].coords[j][1]] = 1

    # Calculate the region proposals
    #lbl = label(prop_map, connectivity=2)

    #props = regionprops(lbl)


    # In[24]:


    # for i in range(len(props)):
    #     print(props[i].centroid)


    # In[30]:


    # Initialize the counters
    tp = 0
    fp = 0
    fn = 0


    # In[31]:

    for i in range(len(props)):

        if (props[i].area < thresh):
            continue

        is_in_bb = False
        min_row = props[i].bbox[0]
        min_col = props[i].bbox[1]
        max_row = props[i].bbox[2]
        max_col = props[i].bbox[3]

        for j in range(len(bb_coords_list)):

            row = (bb_coords_list[j][1] + bb_coords_list[j][3]) / 2
            col = (bb_coords_list[j][0] + bb_coords_list[j][2]) / 2

            if (row >= min_row) and (row <= max_row):

                if (col >= min_col) and (col <= max_col):

                    tp += 1
                    is_in_bb = True

            if not is_in_bb:
                fp += 1

    for i in range(len(bb_coords_list)):

        row = (bb_coords_list[i][1] + bb_coords_list[i][3]) / 2
        col = (bb_coords_list[i][0] + bb_coords_list[i][2]) / 2

        is_there_centroid = False

        for j in range(len(props)):

            if (props[j].area < thresh):
                continue

            min_row = props[j].bbox[0]
            min_col = props[j].bbox[1]
            max_row = props[j].bbox[2]
            max_col = props[j].bbox[3]

            if (row >= min_row) and (row <= max_row):

                if (col >= min_col) and (col <= max_col):

                    is_there_centroid = True

        if not is_there_centroid:
            fn += 1


    # # Count for true positives and false positives
    # # Iterate over all the region proposals
    # for i in range(len(props)):
    #
    #     #if (props[i].area < 200):
    #     #    continue
    #
    #     # Initialize flag for false positive
    #     is_in_bb = False
    #
    #     # Iterate over all the ground truth bounding boxes
    #     for j in range(len(bb_coords_list)):
    #
    #         # Check if it fits in column range
    #         if (bb_coords_list[j][0] <= props[i].centroid[1]) and (bb_coords_list[j][2] >= props[i].centroid[1]):
    #
    #             # Check if it fits in row range
    #             if (bb_coords_list[j][1] <= props[i].centroid[0]) and (bb_coords_list[j][3] >= props[i].centroid[0]) and (props[i].area >= 200):
    #
    #                 tp += 1
    #                 is_in_bb = True
    #                 # break
    #     if not is_in_bb:
    #         fp += 1
    #
    #
    # # In[34]:
    #
    #
    # # Count for false negatives
    # # Iterate over all the ground truth bounding boxes
    # for i in range(len(bb_coords_list)):
    #
    #     is_there_centroid = False
    #
    #     # Iterate over all the regions props
    #     for j in range(len(props)):
    #
    #         if (props[j].area < 200):
    #             continue
    #
    #         # Check if it fits in column range
    #         if (bb_coords_list[i][0] <= props[j].centroid[1]) and (bb_coords_list[i][2] >= props[j].centroid[1]):
    #
    #             # Check if it fits in row range
    #             if (bb_coords_list[i][1] <= props[j].centroid[0]) and (bb_coords_list[i][3] >= props[j].centroid[0]):
    #
    #                 is_there_centroid = True
    #                 break
    #
    #     if not is_there_centroid:
    #         fn += 1

    mycsv.writerow([key, tp, fp, fn, len(bb_coords_list)])
    print(key)
