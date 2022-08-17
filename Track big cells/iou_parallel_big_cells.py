import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
from os import listdir, mkdir
import cv2
import matplotlib.pyplot as plt
from os.path import isfile, join
import numpy as np
from scipy import ndimage
import os
import time
from get_match_big import get_matching
from skimage.measure import label, regionprops
import copy
from skimage import morphology
import argparse

num_cores = multiprocessing.cpu_count()
print(num_cores)
def actual_match(i,mask_dir,gt_dir,dest_dir,onlyfiles,gt_images_list,gt_files,max_id,crop_dim):
    filename = join(mask_dir, onlyfiles[i])
    I2 = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    image_2 = I2
    num1 = np.unique(image_2)
    #image_2[image_2 == num1[-1]] = 0
    mask_compare = np.full(np.shape(image_2), num1[-1])
    separate_mask = np.equal(image_2, mask_compare).astype(int)
    separate_mask[separate_mask>0]=max_id
    image_2[image_2 == num1[-1]] = 0
    regions = regionprops(image_2)
    centroids_2 = [prop.centroid for prop in regions]
    centroids_2 = np.asarray(centroids_2)

    new_image_2 = image_2
    image_num = int(onlyfiles[i][-11:-8])

    nearest = np.argmin(abs(gt_images_list - image_num)) # Get the nearest ground truth image
    filename = join(gt_dir, gt_files[nearest])
    I1 = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    image_1 = I1
    num1 = np.unique(image_1)
    image_1[image_1 == num1[-1]] = 0
    new_image_1 = image_1
    regions = regionprops(image_1)
    centroids_1 = [prop.centroid for prop in regions]
    centroids_1 = np.asarray(centroids_1)

    unique_1 = np.unique(image_1)[1:]
    unique_2 = np.unique(image_2)[1:]

    backward_match = np.zeros((len(unique_2), 2))
    backward_match[:, 0] = unique_2
    
    # For each cell find the cell with maximum overlap in ground truth image
    backward_match = get_matching(new_image_1, new_image_2, crop_dim, unique_2, centroids_2, backward_match)
    # forward_match = get_matching(new_image_2, new_image_1, crop_dim, unique_1, centroids_1, forward_match)
    # print(backward_match)
    new_p2 = new_image_2
    new_patch = np.zeros(new_p2.shape)
    for k in range(len(unique_2)):
        pos1 = np.where(backward_match[:, 0] == unique_2[k])
        if backward_match[pos1, 1] > 0:
            new_patch[new_p2 == backward_match[pos1, 0][0]] = backward_match[pos1, 1][0]
        else:
            new_patch[new_p2 == backward_match[pos1, 0][0]] = max_id

    matched = new_patch+separate_mask

    filename1 = join(dest_dir, onlyfiles[i])
    cv2.imwrite(filename1, matched)



def iou_with_gt(mask_dir, gt_dir,dest_dir):

    isExist = os.path.exists(dest_dir)
    if not isExist:
        os.makedirs(dest_dir)

    onlyfiles = sorted([f for f in listdir(mask_dir) if f.endswith('.png')])
    gt_files = sorted([f for f in listdir(gt_dir) if f.endswith('.tif')])
    nfiles = len(onlyfiles)

    max_id = 30
    start = 0
    stop = nfiles
    crop_dim = [1024,1024]
    my_list =np.arange(start,stop)

    gt_images_list = np.asarray([0,50,100,150,200,300,400,500,600,700])

    processed_list = Parallel(n_jobs=num_cores)(delayed(actual_match)(i, mask_dir,gt_dir,dest_dir,onlyfiles,gt_images_list,gt_files,max_id,crop_dim) for i in range(start, stop))


def main():
    parser = argparse.ArgumentParser(description='EM Segmentation')


    parser.add_argument('--mask_dir', type=str, help='directory containing the predictions')
    parser.add_argument('--gt_dir', type=str, help='directory containing the training masks')
    parser.add_argument('--dest_dir', type=str, help='directory to save files to')

    args = parser.parse_args()

    iou_with_gt(args.mask_dir, args.gt_dir, args.dest_dir)


if __name__ == '__main__':
    main()

