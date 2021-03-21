# # from __future__ import print_function
import numpy as np
def get_random_box(img_size,area_ratio):
    box_w, box_h = (np.sqrt(area_ratio) * np.array(img_size)).astype(np.int32)
    box_cx, box_cy = np.random.randint(0, img_size[0]), np.random.randint(0, img_size[1])
    x1 = max(box_cx - box_w//2,0)
    x2 = min(box_cx + box_w // 2, img_size[0])
    y1 = max(box_cy - box_h//2, 0)
    y2 = min(box_cy + box_h // 2, img_size[1])
    return x1,y1,x2,y2
def cutmix(batch_imgs,one_hot_batch_labels,beta=1,prob=1.):
    if np.random.uniform() < prob:
        batch_size = batch_imgs.shape[0]
        random_batch_indexes = np.random.choice(batch_size, batch_size,replace=False)
        cut_ratio = np.random.beta(beta, beta)
        box = get_random_box(batch_imgs.shape[1:3][::-1], cut_ratio)
        batch_imgs_copy = batch_imgs.copy()
        batch_imgs[:, box[1]:box[3],box[0]:box[2]] = batch_imgs_copy[random_batch_indexes,box[1]:box[3],box[0]:box[2]]
        one_hot_batch_labels = (1.-cut_ratio) * one_hot_batch_labels + cut_ratio * one_hot_batch_labels[random_batch_indexes]
    return batch_imgs,one_hot_batch_labels

