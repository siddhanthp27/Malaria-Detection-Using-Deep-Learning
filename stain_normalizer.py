import os
import cv2
# import staintools
from staintool.brightness_standardizer import BrightnessStandardizer
from staintool.reinhard_color_normalizer import ReinhardColorNormalizer

# One can similarly use the tool for Macenko and Vahadane staining methods

stain_normalizer = ReinhardColorNormalizer()

standardizer = BrightnessStandardizer()

photo = cv2.imread('image_cleaning/3/0d2aba33-6920-4001-bd54-59fe0bf9f50e.png')

file_list = os.listdir('3/')

# file_list = ['8d02117d-6c71-4e47-b50a-6cc8d5eb1d55.png', '6b14c855-8561-417c-97a4-63fa552842fd.png']

for name in file_list:
    img = cv2.imread('image_cleaning/3/'+name)

    photo_standard = standardizer.transform(photo)
    img_standard = standardizer.transform(img)

    #cv2.imwrite('photo_standard.png', photo_standard)
    #cv2.imwrite('img_standard.png', img_standard)

    stain_normalizer.fit(photo_standard)

    img_standard_normalized = stain_normalizer.transform(img_standard)

    cv2.imwrite(name, img_standard_normalized)
    print(name)
