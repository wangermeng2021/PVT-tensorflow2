# # from __future__ import print_function

import numpy as np
def mixup(batch_imgs,one_hot_batch_labels,beta=1,prob=1.):
    if np.random.uniform() < prob:
        batch_size = batch_imgs.shape[0]
        random_batch_indexes = np.random.choice(batch_size, batch_size,replace=False)
        mix_ratio_1 = np.random.beta(beta, beta)
        mix_ratio_2 = 1. - mix_ratio_1
        batch_imgs = mix_ratio_1 * batch_imgs+mix_ratio_2*batch_imgs[random_batch_indexes]
        # print(mix_ratio_1)
        one_hot_batch_labels = mix_ratio_1 * one_hot_batch_labels + mix_ratio_2 * one_hot_batch_labels[random_batch_indexes]
    return batch_imgs,one_hot_batch_labels

