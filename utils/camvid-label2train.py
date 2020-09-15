import os

import pandas as pd
import numpy as np
from PIL import Image

camvid_trainIds2labelIds = np.array(
    [11, 11, 10, 11, 0, 3, 11, 11, 8, 7, 11, 11, 11, 11, 11, 11, 6, 5, 11, 9, 4, 2, 11, 11, 11, 11, 1, 11, 11, 11, 11,
     11],
    dtype=np.uint8)


def LabelID2trainID(trainID_png_dir, save_dir):
    print('save_dir:  ', save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    png_list = os.listdir(trainID_png_dir)
    for index, png_filename in enumerate(png_list):
        #
        png_path = os.path.join(trainID_png_dir, png_filename)
        if png_path.split('.')[0][-1] != 'P':
            continue
        # print(png_path)
        print('processing(', index, '/', len(png_list), ') ....')
        image = Image.open(png_path)  # image is a PIL #image
        pngdata = np.array(image)
        trainID = pngdata  # model prediction
        row, col = pngdata.shape
        labelID = np.zeros((row, col), dtype=np.uint8)
        for i in range(row):
            for j in range(col):
                labelID[i][j] = camvid_trainIds2labelIds[trainID[i][j]]

        png_filename = png_filename.split('.')[0] + '_P.png'
        res_path = os.path.join(save_dir, png_filename)
        new_im = Image.fromarray(labelID)
        new_im.save(res_path)


if __name__ == '__main__':
    trainID_png_dir = 'dataset/camvid/labels'
    save_dir = 'dataset/camvid/labels'
    LabelID2trainID(trainID_png_dir, save_dir)
