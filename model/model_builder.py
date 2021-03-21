
from model.efficientnet import EfficientNet
from model.resnet import ResNet
from model.PVT import PVTNet
from model.VIT import VITNet
from model.resnet_for_cifar import ResNetForCifar
def get_model(args,num_class):
    if args.model[0:3] == "Res":
        try:
            depth = int(args.model[-3:])
        except:
            depth = int(args.model[-2:])
        model = ResNet(classes=num_class,type=depth,  concat_max_and_average_pool=args.concat_max_and_average_pool,
                       pretrain=args.pretrain)
    elif args.model[0:3] == "Eff":
        model = EfficientNet(classes=num_class,type=args.model[-2:], concat_max_and_average_pool=args.concat_max_and_average_pool,
                       pretrain=args.pretrain)
    elif args.model[0:3] == "PVT":
        model = PVTNet(img_size=args.img_size, classes=num_class,type=args.model.split('-')[-1],pretrain=args.pretrain)
    elif args.model[0:5] == "Cifar":
        model = ResNetForCifar(classes=num_class, concat_max_and_average_pool=args.concat_max_and_average_pool,pretrain=args.pretrain)
    else:
        raise ValueError("{} is not supported!".format(args.model))
    return model.get_model()
