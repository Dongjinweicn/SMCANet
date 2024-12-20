from argparse import ArgumentParser


def add_experiment_args(parser: ArgumentParser) -> None:
    # define task, label values, and output channels
    tasks = {
        # 'MR': {'lab_values': [0, 600, 200, 500], 'out_channels': 4}
        'MR': {'lab_values': [0, 1, 2, 3, 4, 5], 'out_channels': 6}
    }

    # Experiment
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--lr_drop', default=500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--tasks', default=tasks, type=dict)

    # Model parameters
    parser.add_argument('--model', default='SMCAnet', required=False)
    # parser.add_argument('--dataset', default='bayesian_ZS_run_scar', type=str,
    #                     help='multi-sequence CMR segmentation dataset')
    # parser.add_argument('--dataset', default='centerB', type=str,
    parser.add_argument('--dataset', default='CARE2024_MyoPS', type=str,
                        help='multi-sequence CMR segmentation dataset')
    parser.add_argument('--sequence', default='test_3modalities', type=str,
                        help='which CMR sequence')
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--in_channels', default=3, type=int)

    # loss weight
    parser.add_argument('--CrossEntropy_loss_coef', default=1, type=float)
    parser.add_argument('--Inclusive_loss_coef', default=1, type=float)
    parser.add_argument('--AvgDice_loss_coef', default=-1, type=float)
    parser.add_argument('--Bayes_loss_coef', default=100, type=float)


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--output_dir', default='./logs/model',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    parser.add_argument('--eval', default=False, action='store_true')
    # parser.add_argument('--eval', default=True, action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


def add_bayes_args(parser: ArgumentParser) -> None:
    # prior hyper-params
    parser.add_argument('--mu_0', default=0, type=float)








