from .mscmr_V2 import build


def build_dataset(image_set, args):
    if args.dataset == 'test':
        print(f'{args.dataset}')
        return build(image_set, args)
    raise ValueError(f'dataset {args.dataset} not supported')
