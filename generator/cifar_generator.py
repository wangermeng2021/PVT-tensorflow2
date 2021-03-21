
import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
from utils.snapmix import snapmix
from utils.cutmix import cutmix
from utils.mixup import mixup
import albumentations as A
from utils.custom_augment import train_augment
from utils.preprocess import normalize
from utils import auto_augment,rand_augment
class CIFARGenerator(tf.keras.utils.Sequence):
    def __init__(self, args, mode='train'):

        self.args = args
        root_dir = args.dataset
        batch_size = args.batch_size
        augment = args.augment
        self.resize_size = (40,  40)
        self.crop_size = (32, 32)

        if args.dataset.strip()[5:]=='10':
            self.num_class = 10
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
        elif args.dataset.strip()[5:]=='100':
            self.num_class = 100
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
        else:
            raise ValueError('unsupported dataset:{}'.format(args.dataset))


        if mode == 'train':
            self.imgs = train_images
            self.labels = train_labels
            self.data_index = np.arange(0, len(self.imgs))
            np.random.shuffle(self.data_index)
        else:
            self.imgs = test_images
            self.labels = test_labels
            self.data_index = np.arange(0, len(self.imgs))

        self.augment = augment
        self.mode = mode
        self.batch_size = batch_size
        self.eppch_index = 0


        self.train_transform = A.Compose([
            A.RandomCrop(width=self.crop_size[0], height=self.crop_size[1]),
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
        ])

    def resize(self, img, size):
        size = np.array(size)
        img_width_height = img.shape[0:2][::-1]
        max_ratio = np.max(img_width_height / size)
        img_width_height = img_width_height / max_ratio
        min_side = np.min(img_width_height).astype(np.int32)
        dst_size = [[min_side, size[1]], [size[0], min_side]][np.argmin(img_width_height)]
        resized_img = cv2.resize(img, tuple(dst_size))
        pad_height = (size[1] - resized_img.shape[0]) // 2
        pad_width = (size[0] - resized_img.shape[1]) // 2

        # cv2.copyMakeBorder(resized_img,pad_height, size[0] - pad_height - resized_img.shape[0], pad_width, size[1] - pad_width - resized_img.shape[1],cv2.BORDER_REPLICATE);

        return np.pad(resized_img, [[pad_height, size[1] - pad_height - resized_img.shape[0]],
                                    [pad_width, size[0] - pad_width - resized_img.shape[1]], [0,0]],constant_values=np.random.randint(0, 255))
    def read_img(self, path):
        image = np.ascontiguousarray(Image.open(path).convert('RGB'))
        return image
        # return image[:, :, ::-1]
    def on_epoch_end(self):
        if self.mode == 'train':
            np.random.shuffle(self.data_index)
        self.eppch_index += 1
    def __len__(self):
        return int(np.ceil(len(self.imgs) / self.batch_size))
    def __getitem__(self, batch_index):
        cur_batch_imgs = self.imgs[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        cur_batch_labels = self.labels[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        one_hot_batch_labels = np.zeros([len(cur_batch_imgs), self.num_class])
        batch_imgs = []
        if self.mode == "valid":
            for i in range(len(cur_batch_imgs)):
                img = cur_batch_imgs[i]
                if self.args.pretrain:
                    if self.args.model[0:3] == "Res":
                        img = normalize(img, mode='caffe')
                else:
                    img = normalize(img, mode='tf')
                batch_imgs.append(img)
                one_hot_batch_labels[i, cur_batch_labels[i]] = 1
            batch_imgs = np.array(batch_imgs)
        else:
            for i in range(len(cur_batch_imgs)):
                img = cur_batch_imgs[i]
                img = np.pad(img,((4,4),(4,4),(0,0)))
                img = self.train_transform(image=img)['image']

                if self.args.pretrain:
                    if self.args.model[0:3] == "Res":
                        img = normalize(img,mode='caffe')
                else:
                    img = normalize(img, mode='tf')
                batch_imgs.append(img)
                one_hot_batch_labels[i, cur_batch_labels[i]] = 1
            batch_imgs = np.array(batch_imgs)

            # # # if np.random.rand(1) < 0.5:
            # if self.augment == 'cutmix':
            #     batch_imgs, one_hot_batch_labels = cutmix(batch_imgs,one_hot_batch_labels,3)
            # elif self.augment == 'mixup':
            #     batch_imgs, one_hot_batch_labels = mixup(batch_imgs,one_hot_batch_labels,1)

        return batch_imgs, one_hot_batch_labels




