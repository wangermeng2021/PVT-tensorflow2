# # from __future__ import print_function
import numpy as np
import cv2

def get_cam(conv_out_ori,dense_layer,labels,img_size):
    filters, biases = dense_layer.get_weights()
    conv_out = np.reshape(conv_out_ori, (conv_out_ori.shape[0], -1, conv_out_ori.shape[-1]))
    filters = np.squeeze(filters)
    out1 = np.matmul(conv_out, filters)
    out2 = out1 + biases
    cam_out = np.reshape(out2, [np.shape(out2)[0], np.shape(conv_out_ori)[1], np.shape(conv_out_ori)[2], np.shape(out2)[-1]]).astype(np.float32)
    car_cam_list = []
    for i in range(labels.shape[0]):
        resized_cam_out = cam_out[i, :, :, labels[i]]
        resized_cam_out = cv2.resize(resized_cam_out,img_size,interpolation=cv2.INTER_LINEAR)
        car_cam_list.append(resized_cam_out)
    car_cam = np.stack(car_cam_list)

    s1 = np.reshape(np.min(car_cam,axis=(1,2)),[car_cam.shape[0],1,1])
    s1 = car_cam - s1
    s2 = np.sum(s1, axis=(1, 2))
    s2 = np.reshape(s2, [s2.shape[0],1,1])
    car_cam = s1/s2

    return car_cam

