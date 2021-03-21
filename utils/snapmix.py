# # from __future__ import print_function
import cv2
import numpy as np
from utils.cam import get_cam
def get_random_box(img_size, area_ratio):
    # area_ratio = 1. - area_ratio
    side_length_ratio = np.sqrt(area_ratio)
    side_length = side_length_ratio*img_size
    side_length = side_length.astype(np.int32)

    half_side_length = side_length//2
    w_offset, h_offset = np.random.randint(0,img_size[0]), np.random.randint(0,img_size[1])
    x1 = max(w_offset - half_side_length[0], 0)
    y1 = max(h_offset - half_side_length[1], 0)
    x2 = min(w_offset + half_side_length[0], img_size[0])
    y2 = min(h_offset + half_side_length[1], img_size[1])
    return x1,y1,x2,y2

def snapmix(batch_imgs,batch_one_hot_labels,last_conv_model,last_dense_layer,img_size,beta):
    conv_out = last_conv_model.predict(batch_imgs)
    batch_labels = np.argmax(batch_one_hot_labels, axis=-1)
    car_cam = get_cam(conv_out, last_dense_layer, batch_labels, img_size)

    img_size = np.array(img_size)
    beta1 = np.random.beta(beta,beta)
    beta2 = np.random.beta(beta, beta)

    box1 = get_random_box(img_size, beta1)
    box2 = get_random_box(img_size, beta2)
    num_batch = np.shape(batch_imgs)[0]
    random_batch_indexs = np.random.choice(num_batch,num_batch,replace=False)

    same_class_mask = batch_labels == batch_labels[random_batch_indexs]

    weight1 = 1. - np.sum(car_cam[:,box1[1]:box1[3],box1[0]:box1[2]],axis=(1,2))/(np.sum(car_cam,axis=(1,2))+1e-8)
    weight2 = np.sum(car_cam[random_batch_indexs, box2[1]:box2[3], box2[0]:box2[2]], axis=(1, 2)) / (np.sum(car_cam[random_batch_indexs], axis=(1, 2))+1e-8)
    weight1_copy = weight1.copy()
    weight1[same_class_mask] += weight2[same_class_mask]
    weight2[same_class_mask] += weight1_copy[same_class_mask]

    weight1 = np.reshape(weight1, [weight1.shape[0], 1])
    weight2 = np.reshape(weight2, [weight2.shape[0], 1])

    batch_one_hot_labels_copy = batch_one_hot_labels.copy()
    batch_one_hot_labels = weight1*batch_one_hot_labels+weight2*batch_one_hot_labels_copy[random_batch_indexs]
    batch_imgs_copy = batch_imgs.copy()
    for bi in range(num_batch):
        batch_imgs[:,box1[1]:box1[3],box1[0]:box1[2]][bi] = \
            cv2.resize(batch_imgs_copy[random_batch_indexs, box2[1]:box2[3], box2[0]:box2[2]][bi],(box1[2]-box1[0],box1[3]-box1[1]), interpolation=cv2.INTER_LINEAR)

    return batch_imgs,batch_one_hot_labels
