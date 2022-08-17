
import numpy as np

def get_matching(new_image_1,new_image_2,crop_dim,unique_2,centroids_2,match_matrix):
    """ For every region in new_image_2 find a region with maximum Intersection over Union (IoU) in new_image_1.
    """
    for crop_cc in range(len(unique_2)):
        cur_cent = np.floor(centroids_2[crop_cc]).astype(int)

        crop_up = cur_cent[0] - crop_dim[0] if cur_cent[0] - crop_dim[0] > 0 else 0
        crop_down = cur_cent[0] + crop_dim[1] if cur_cent[0] + crop_dim[1] < new_image_1.shape[0] else new_image_1.shape[0]

        crop_left = cur_cent[1] - crop_dim[0] if cur_cent[1] - crop_dim[0] > 0 else 0
        crop_right = cur_cent[1] + crop_dim[1] if cur_cent[1] + crop_dim[1] < new_image_1.shape[1] else new_image_1.shape[1]

        # crop_labels_1 = new_image_1[crop_left:crop_right, crop_up:crop_down]
        # crop_labels_2 = new_image_2[crop_left:crop_right, crop_up:crop_down]
        crop_labels_1 = new_image_1[crop_up:crop_down, crop_left:crop_right]
        crop_labels_2 = new_image_2[crop_up:crop_down, crop_left:crop_right]

        u_1 = np.unique(crop_labels_1)[1:]
        n_1 = len(u_1)

        if n_1 > 0:
            a = crop_labels_1

            mask_compare = np.full(np.shape(crop_labels_2), unique_2[crop_cc])
            b = np.equal(crop_labels_2, mask_compare).astype(int)
            a_n = np.zeros((np.shape(a)[0], np.shape(a)[1], n_1))
            for j in range(n_1):
                a_n[:, :, j] = np.equal(a, np.full(np.shape(a), u_1[j])).astype(int)
            a_n = a_n.astype(int)

            b_n_rep = np.repeat(b[:, :, np.newaxis], n_1, axis=2)

            nume = b_n_rep * a_n
            den = (b_n_rep | a_n)
            den = den.astype(int)
            new_num = nume.reshape(nume.shape[0] * nume.shape[1], nume.shape[2])
            new_den = den.reshape(nume.shape[0] * nume.shape[1], nume.shape[2])
            num_nz = np.count_nonzero(new_num, axis=0)
            den_nz = np.count_nonzero(new_den, axis=0)
            iou = (num_nz / den_nz)

            max_col = np.amax(iou, axis=0)
            ind_col = np.argmax(iou, axis=0)

            if max_col > 0.00001:
                pos1 = np.where(match_matrix[:, 0] == unique_2[crop_cc])
                match_matrix[pos1, 1] = u_1[ind_col]
    return match_matrix
