
import tensorflow as tf
import tensorflow_addons as tfa

def get_optimizers(args):
    if args.optimizer == 'sgd':
        optimizer = tfa.optimizers.SGDW(learning_rate=args.init_lr,momentum=args.momentum,name='sgd',weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = tfa.optimizers.AdamW(learning_rate=args.init_lr,name='adam',weight_decay=args.weight_decay)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=args.init_lr, name='adam')
    else:
        raise ValueError("{} is not supported!".format(args.optimizer))
    return optimizer

