from .segmentation import build as build_Unet
from .SMCAnet import build as build_SMCAnet

def build_model(args):
    if args.model == 'Unet':
        return build_Unet(args)
    elif args.model == 'SMCAnet':
        return build_SMCAnet(args)
    else:
        raise ValueError('invalid model:{}'.format(args.model))
