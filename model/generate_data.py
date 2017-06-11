import utils
import numpy as np
import os
import skimage
image_path = '../dataset/dogcat/'
output_path = '../data/'
downsa_img = list()
resa_img = list()

counter = 0
max_num = 4200

for img_name in os.listdir(image_path):
    counter += 1
    print(counter)
    downsample, resample = utils.transfer_image(image_path + img_name)
    downsa_img.append(downsample)
    print('down', np.max(downsample))
    resa_img.append(resample)
    print('re', np.max(resample))
    if counter > max_num:
        break

downsa_img = np.asarray(downsa_img)
np.save(output_path+'downsa_img', downsa_img)
del downsa_img
resa_img = np.asarray(resa_img)
np.save(output_path+'resa_img', resa_img)




