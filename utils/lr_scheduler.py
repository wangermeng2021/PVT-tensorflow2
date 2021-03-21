
import tensorflow as tf
import numpy as np
def get_lr_scheduler(args):
    if args.lr_scheduler == 'step':
        def scheduler(epoch,lr=0.001):
            if epoch < args.warmup_epochs:
                current_epoch_lr = args.warmup_lr + epoch * (args.init_lr -args.warmup_lr) / args.warmup_epochs
                print("epoch:{} lr:{:.6f}".format(epoch,current_epoch_lr.numpy()))
                return current_epoch_lr
            else:
                for index, val in enumerate(args.lr_decay_epoch):
                    if epoch < val:
                        current_epoch_lr = args.init_lr*args.lr_decay**index
                        print("epoch:{} lr:{:.6f}".format(epoch,current_epoch_lr.numpy()))
                        return current_epoch_lr
            current_epoch_lr = args.init_lr*args.lr_decay**len(args.lr_decay_epoch)
            print("epoch:{} lr:{:.6f}".format(epoch,current_epoch_lr.numpy()))
            return current_epoch_lr
        return scheduler
    elif args.lr_scheduler == 'cosine':
        def scheduler(epoch,lr=0.001):
            if epoch < args.warmup_epochs:
                current_epoch_lr = args.warmup_lr + epoch * (args.init_lr - args.warmup_lr) / args.warmup_epochs
            else:
                current_epoch_lr = args.init_lr * (
                        1.0 + tf.math.cos(np.pi / (args.epochs - args.warmup_epochs) * (epoch - args.warmup_epochs))) / 2.0
            print("epoch:{} lr:{:.6f}".format(epoch,current_epoch_lr.numpy()))
            return current_epoch_lr
    else:
        raise ValueError("{} is not supported!".format(args.lr_scheduler))
    return scheduler

