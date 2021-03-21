
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
class CustomGenerator(tf.keras.utils.Sequence):
    def __init__(self, args, mode='train'):
        self.args = args
        root_dir = args.dataset
        batch_size = args.batch_size
        augment = args.augment
        img_size = args.img_size

        self.resize_size = ( int(img_size/0.875),  int(img_size/0.875))
        self.crop_size = (img_size, img_size)

        self.root_dir = os.path.join(root_dir,mode)
        self.img_path_list = []
        self.label_list = []
        self.class_names = []
        for class_index,class_name in enumerate(os.listdir(self.root_dir)):
            class_dir = os.path.join(self.root_dir,class_name)
            for img_name in os.listdir(class_dir):
                self.img_path_list.append(os.path.join(class_dir,img_name))
                self.label_list.append(class_index)
            self.class_names.append(class_name)

        if mode == 'train':
            pad_len = len(self.img_path_list)%batch_size
            img_path_list_len = len(self.img_path_list)
            for _ in range(pad_len):
                rand_index = np.random.randint(0,img_path_list_len)
                self.img_path_list.append(self.img_path_list[rand_index])
                self.label_list.append(self.label_list[rand_index])
            self.data_index = np.arange(0, len(self.label_list))
            np.random.shuffle(self.data_index)
        else:
            self.data_index = np.arange(0, len(self.label_list))
        self.img_path_list = np.array(self.img_path_list)
        self.label_list = np.array(self.label_list)
        self.augment = augment
        self.mode = mode
        self.batch_size = batch_size
        self.eppch_index = 0
        self.num_class = len(self.class_names)

        self.random_crop_transform = A.Compose([
            A.RandomCrop(width=self.crop_size[0], height=self.crop_size[1]),
        ])
        self.center_crop_transform = A.Compose([
            A.CenterCrop(width=self.crop_size[0], height=self.crop_size[1]),
        ])

        self.auto_augment = auto_augment.AutoAugment()
        self.rand_augment = rand_augment.RandAugment(magnitude=10.)

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

        return np.pad(resized_img, [[pad_height, size[0] - pad_height - resized_img.shape[0]],
                                    [pad_width, size[1] - pad_width - resized_img.shape[1]], [0,0]],constant_values=np.random.randint(0, 255))
    def read_img(self, path):
        image = np.ascontiguousarray(Image.open(path).convert('RGB'))
        return image
        # return image[:, :, ::-1]
    def on_epoch_end(self):
        if self.mode == 'train':
            np.random.shuffle(self.data_index)
        self.eppch_index += 1
    def __len__(self):
        return int(np.ceil(len(self.img_path_list) / self.batch_size))
    def __getitem__(self, batch_index):

        batch_img_paths = self.img_path_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        batch_labels = self.label_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        one_hot_batch_labels = np.zeros([len(batch_img_paths), self.num_class])
        batch_imgs = []
        if self.mode == "valid":
            for i in range(len(batch_img_paths)):
                img = self.read_img(batch_img_paths[i])
                img = self.resize(img, self.resize_size)
                img = self.center_crop_transform(image=img)['image']

                if self.args.pretrain:
                    if self.args.model[0:3] == "Res":
                        img = normalize(img, mode='caffe')
                else:
                    img = normalize(img, mode='tf')

                batch_imgs.append(img)
                one_hot_batch_labels[i, batch_labels[i]] = 1
            batch_imgs = np.array(batch_imgs)
        else:
            for i in range(len(batch_img_paths)):
                img = self.read_img(batch_img_paths[i]).astype(np.uint8)

                img = self.resize(img, self.resize_size)
                img = self.random_crop_transform(image=img)['image']

                if self.augment == 'auto_augment':
                    img = self.auto_augment.distort(tf.constant(img)).numpy()
                elif self.augment == 'rand_augment':
                    img = self.rand_augment.distort(tf.constant(img)).numpy()
                elif self.augment == 'custom_augment':
                    img = train_augment(img)
                # print(self.resize_size,self.crop_size)
                # cv2.imshow("db",img)
                # cv2.waitKey()
                if self.args.pretrain:
                    if self.args.model[0:3] == "Res":
                        img = normalize(img,mode='caffe')
                else:
                    img = normalize(img, mode='tf')
                    pass
                batch_imgs.append(img)
                one_hot_batch_labels[i, batch_labels[i]] = 1
            batch_imgs = np.array(batch_imgs)
            # # if np.random.rand(1) < 0.5:
            if self.augment == 'cutmix':
                batch_imgs, one_hot_batch_labels = cutmix(batch_imgs,one_hot_batch_labels,3)
            elif self.augment == 'mixup':
                batch_imgs, one_hot_batch_labels = mixup(batch_imgs,one_hot_batch_labels,1)

        return batch_imgs, one_hot_batch_labels




