

from generator.custom_generator import CustomGenerator
from generator.cifar_generator import CIFARGenerator
def get_generator(args):
    if args.dataset.strip()[0:5] == 'cifar':
        train_generator = CIFARGenerator(args, mode="train")
        val_generator = CIFARGenerator(args, mode="valid")
        return train_generator, val_generator
    else:
        train_generator = CustomGenerator(args, mode="train")
        val_generator = CustomGenerator(args, mode="valid")
        return train_generator, val_generator
