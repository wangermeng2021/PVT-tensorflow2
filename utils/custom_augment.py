
import albumentations as A
import cv2
def train_augment(img):
    transform = A.Compose([
        # add your custom augmentment,....
        A.HorizontalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.2),

    ])
    img = transform(image=img)['image']
    return img