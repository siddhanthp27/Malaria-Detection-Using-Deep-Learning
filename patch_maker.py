import cv2
import torchvision.transforms as transform
import numpy as np
import os
import xml.etree.ElementTree as ET
# from heatmap import add
import matplotlib.pyplot as plt
import json
import pickle
from pprint import pprint
from random import shuffle

image_bb = pickle.load(open('image_bb_dict.pkl', 'rb'))

categories = set()

categories.add(u'background')

keys = image_bb.keys()

data = json.load(open('training.json'))

categories = set()

categories.add(u'background')

for i in range(len(data)):
    for j in range(len(data[i]['objects'])):
        categories.add(data[i]['objects'][j]['category'])

category_dict = dict()

for i, category in enumerate(categories):
    category_dict[category] = float(i)

file_name = os.listdir('3/')

for file_index in range(len(file_name)):

    pathname = '3/' + file_name[file_index]

    print(pathname)

    img = cv2.imread(pathname)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    bbs = image_bb[file_name[file_index]]

    # print(img)

    cnt = 0

    for i in range(len(bbs)):
       ymin = bbs[i]['bounding_box']['minimum']['c']
       xmin = bbs[i]['bounding_box']['minimum']['r']
       ymax = bbs[i]['bounding_box']['maximum']['c']
       xmax = bbs[i]['bounding_box']['maximum']['r']

       c = category_dict[bbs[i]['category']]

       if c == 0.0 or c == 3.0 or c == 4.0 or c == 6.0:
           xmin = max(40, xmin)
           ymin = max(40, ymin)
           xmax = min(1200-40, xmax)
           ymax = min(1600-40, ymax)
           x = int((xmin + xmax) / 2)
           y = int((ymin + ymax) / 2)
           #points = [(i, j) for i in range(xmin, xmax) for j in range(ymin, ymax)]

           #shuffle(points)

           #chosen_points = []

           #while(len(chosen_points) < 10):
            #   x = points[0][0]
            #   y = points[0][1]
            #   if x-55>=0 and x+56<1200 and y-55>=0 and y+56<1600:
            #       chosen_points.append((x, y))
            #   points.pop(0)

           #for x, y in chosen_points:

           patch_img = img[x-40:x+40, y-40:y+40, :]

           patch_gen_name = file_name[file_index].split('.')[0]

           patch_name = 'patches/malaria/' + patch_gen_name + '_patch_' + str(cnt) + '.png'

           cv2.imwrite(patch_name, patch_img)

           cnt += 1

        #    patch_name = 'patches/malaria/' + patch_gen_name + '_patch_' + str(cnt) + '.png'
           #
        #    flip_img = cv2.flip(patch_img, 0)
           #
        #    cv2.imwrite(patch_name, flip_img)
           #
        #    cnt += 1
           #
        #    patch_name = 'patches/malaria/' + patch_gen_name + '_patch_' + str(cnt) + '.png'
           #
        #    flip_img = cv2.flip(patch_img, 1)
           #
        #    cv2.imwrite(patch_name, flip_img)
           #
        #    cnt += 1

           patch_name = 'patches/malaria/' + patch_gen_name + '_patch_' + str(cnt) + '.png'

           flip_img = cv2.flip(patch_img, -1)

           cv2.imwrite(patch_name, flip_img)

           cnt += 1

    cnt = 0

    for i in range(len(bbs)):
        ymin = bbs[i]['bounding_box']['minimum']['c']
        xmin = bbs[i]['bounding_box']['minimum']['r']
        ymax = bbs[i]['bounding_box']['maximum']['c']
        xmax = bbs[i]['bounding_box']['maximum']['r']

        c = category_dict[bbs[i]['category']]

        if c == 2.0:
            xmin = max(40, xmin)
            ymin = max(40, ymin)
            xmax = min(1200-40, xmax)
            ymax = min(1600-40, ymax)
            x = int((xmin + xmax) / 2)
            y = int((ymin + ymax) / 2)
            #points = [(i, j) for i in range(xmin, xmax) for j in range(ymin, ymax)]

            #shuffle(points)

            #chosen_points = []

            #while(len(chosen_points) < 10):
             #   x = points[0][0]
             #   y = points[0][1]
             #   if x-55>=0 and x+56<1200 and y-55>=0 and y+56<1600:
             #       chosen_points.append((x, y))
             #   points.pop(0)

            #for x, y in chosen_points:

            patch_img = img[x-40:x+40, y-40:y+40, :]

            patch_gen_name = file_name[file_index].split('.')[0]

            patch_name = 'patches/rbcs/' + patch_gen_name + '_patch_' + str(cnt) + '.png'

            cv2.imwrite(patch_name, patch_img)

            cnt += 1

    class_map = np.full((1600, 1200), 0)

    for i in range(len(data[file_index]['objects'])):
        ymin = data[file_index]['objects'][i]['bounding_box']['minimum']['c']
        xmin = data[file_index]['objects'][i]['bounding_box']['minimum']['r']
        ymax = data[file_index]['objects'][i]['bounding_box']['maximum']['c']
        xmax = data[file_index]['objects'][i]['bounding_box']['maximum']['r']

        for j in range(ymin, ymax):
            for k in range(xmin, xmax):
                c = category_dict[data[file_index]['objects'][i]['category']]

                if c == 0.0 or c == 2.0 or c == 3.0 or c == 4.0 or c == 6.0:
                    class_map[j][k] = 1

    points = []

    for i in range(40, 1200-40):
        for j in range(40, 1600-40):
            points.append((i, j))

    shuffle(points)

    chosen_points = []

    while(len(chosen_points) < 10):
        x = points[0][0]
        y = points[0][1]
        if (class_map[y][x] == 0):
            chosen_points.append((y, x))

        points.pop(0)

    for i in range(len(chosen_points)):
        y = chosen_points[i][0]
        x = chosen_points[i][1]

        patch_img = img[x-40:x+40, y-40:y+40, :]

        patch_gen_name = file_name[file_index].split('.')[0]

        patch_name = 'patches/non-malaria/' + patch_gen_name + '_patch_' + str(i) + '.png'

        cv2.imwrite(patch_name, patch_img)
