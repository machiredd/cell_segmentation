from os import listdir, mkdir
import cv2
import matplotlib.pyplot as plt
from os.path import isfile, join
import numpy as np
from scipy import ndimage
import os
import time
from get_match import get_matching
import multiprocessing
from skimage.measure import label, regionprops
import copy

start_time1 = time.time()
dest_direc = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM2/101a/track_small/results/per_frame_aligned_3/'
isExist = os.path.exists(dest_direc)
if not isExist:
    os.makedirs(dest_direc)

temp_direc = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM2/101a/track_small/results/per_frame_temp4/'
#temp_direc = '/home/exacloud/gscratch/gray_lab/machired/new_temp1/'
#temp_direc = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM2/101a/track/data/py_temp/'
isExist = os.path.exists(temp_direc)
if not isExist:
    os.makedirs(temp_direc)

mask_direc = '/home/groups/graylab_share/OMERO.rdsStore/machired/EM2/101a/track_big/process/results/per_frame_aligned/'
onlyfiles = sorted([f for f in listdir(mask_direc) if f.endswith('.png')])
nfiles = len(onlyfiles)

img = 0
new_max = 1
start = 1
stop = nfiles
num = np.zeros(stop - start)
crop_dim = [256,256] # crop size within which to compare overlap of regions
island_no = 30

for i in range(start, stop):
    start_time = time.time()
    print(i)
    if i == start: # Give unique label to each region in the first image
        filename = join(mask_direc, onlyfiles[i - 1])
        I1 = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
        image_1 = I1
        #image_1 = I1[1:2000, 1:2000]

        mask_compare = np.full(np.shape(image_1), island_no)
        separate_mask = np.equal(image_1, mask_compare).astype(int)

        separate_mask = separate_mask.astype(np.uint8)
        (numLabels_1, labels_1, stats_1, centroids_1) = cv2.connectedComponentsWithStats(separate_mask, connectivity=8,
                                                                                         ltype=cv2.CV_32S)
        image_1[image_1 == island_no] = 0

        labels_1 = labels_1 + island_no
        labels_1[labels_1 == island_no] = 0
        regions = regionprops(labels_1)
        centroids_1 = [prop.centroid for prop in regions]
        centroids_1 = np.asarray(centroids_1)

        new_image_1 = image_1 + labels_1
        new_max = np.amax(new_image_1)
        max_id = new_max + 1

        save_name = temp_direc + onlyfiles[i-1] + '.txt'
        np.savetxt(save_name, new_image_1, fmt='%d')

    # Get labels of all regions in the next image
    filename = join(mask_direc, onlyfiles[i])
    I2 = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    image_2 = I2
    mask_compare = np.full(np.shape(image_2), island_no)
    separate_mask = np.equal(image_2, mask_compare).astype(int)

    separate_mask = separate_mask.astype(np.uint8)
    (numLabels_2, labels_2, stats_2, centroids_2) = cv2.connectedComponentsWithStats(separate_mask, connectivity=8,
                                                                                     ltype=cv2.CV_32S)
    labels_2 = labels_2 + island_no
    labels_2[labels_2 == island_no] = 0

    regions = regionprops(labels_2)
    centroids_2 = [prop.centroid for prop in regions]
    centroids_2 = np.asarray(centroids_2)

    image_2[image_2 == island_no] = 0
    new_image_2 = labels_2 + image_2

    unique_1 = np.unique(labels_1)[1:]
    unique_2 = np.unique(labels_2)[1:]

    forward_match = np.zeros((len(unique_1), 2))
    backward_match = np.zeros((len(unique_2), 2))
    forward_match[:, 0] = unique_1
    backward_match[:, 0] = unique_2

    # Calculate Intersection over Union (IoU) between regions in two images
    backward_match = get_matching(new_image_1, new_image_2, crop_dim, unique_2, centroids_2, backward_match)
    forward_match = get_matching(new_image_2, new_image_1, crop_dim, unique_1, centroids_1, forward_match)

    # Assign labels from the previous images to overlapping regions
    new_p2 = new_image_2
    new_patch = np.zeros(new_p2.shape)
    for k in range(len(unique_2)):
        pos1 = np.where(backward_match[:, 0] == unique_2[k])
        if backward_match[pos1, 1] > 0:
            new_patch[new_p2 == backward_match[pos1, 0][0]] = backward_match[pos1, 1][0]
        else:
            new_patch[new_p2 == backward_match[pos1, 0][0]] = max_id # No overlap assign new label
            max_id = max_id + 1
    matched = new_patch + image_2
    # matched[:, :, img + 1] = new_patch + image_2

    save_name = temp_direc + onlyfiles[i] +'.txt'
    np.savetxt(save_name, matched, fmt='%d')

    if i == start:
        matching = forward_match
    else:
        for k in range(len(forward_match)):
            if forward_match[k,0] not in matching[:,0]:
                matching = np.vstack([matching, forward_match[k,:]])
            else:
                alre = np.where(matching[:, 0] == forward_match[k,0])
                val = forward_match[k,1]
                if (val < island_no) and (val > 0):
                    matching[alre, 1] = val

    img = img + 1
    new_image_1 = matched
    labels_1 = copy.deepcopy(new_image_1)
    labels_1[labels_1<island_no] = 0
    labels_1=labels_1.astype(np.int64)
    #labels_1=labels_1.astype(np.uint8)
    regions = regionprops(labels_1)
    centroids_1 = [prop.centroid for prop in regions]
    centroids_1 = np.asarray(centroids_1)

    end_time = time.time()
    print(f"Runtime for this image is {end_time - start_time}")

matching = np.asarray(matching)

ind_1 = np.where(matching[:, 1] == 0)
ind_2 = np.where(matching[:, 1] > island_no)
ind_3 = np.where(matching[:, 1] == matching[:, 0])
ind = np.concatenate((ind_1, ind_2, ind_3), axis=1)
ind = np.unique(ind)
l = np.full(matching.shape[0], False)
l[ind] = True
new_match = matching[~l, :]

# print(matching)
# print(new_match)

k = 0
for i in range(start-1, stop):
    #filename = join(mask_direc, onlyfiles[i])
    #I1 = cv2.imread(filename, cv2.COLOR_BGR2GRAY)
    #I2 = I1

    load_name = temp_direc+ onlyfiles[i]+'.txt'
    sel = np.loadtxt(load_name, dtype=int)
    os.remove(load_name)
    # sel = matched[:, :, k]
    present = np.in1d(new_match[:, 0], sel.flatten())
    ind_1 = np.where(present == True)[0]
    if ind_1.size > 0:
        for j in range(len(ind_1)):
            sel[sel == new_match[ind_1[j], 0]] = new_match[ind_1[j], 1]
    sel[sel > island_no] = island_no

    I2 = sel
    #I2[1:2000, 1:2000] =sel
    #plt.imshow(I2[888:1157, 4698:5400])
    #plt.show()
    filename1 = join(dest_direc, onlyfiles[i])
    cv2.imwrite(filename1, I2)
    k = k + 1


end_time1 = time.time()
print(f"Runtime of the program is {end_time1 - start_time1}")
np.savetxt(dest_direc +'matching_full.csv', matching, delimiter=",")
np.savetxt(dest_direc +'new_match.csv', new_match, delimiter=",")
