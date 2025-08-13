import configargparse


def get_opts(args=None):
    parser = configargparse.ArgumentParser()

    if args:
        return parser.parse_args(args)

    # configure file
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')

    # dataset
    parser.add_argument('--dataset_dir', type=str, default="/dataset/c3vd_train/c3vd/train")
    parser.add_argument('--dataset_name', type=str,
                        default='c3vd', choices=['c3vd', 'SCARED'])
    parser.add_argument('--sequence_length', type=int,
                        default=3, help='number of images for training')
    parser.add_argument('--skip_frames', type=int, default=3,
                        help='jump sampling from video')
    parser.add_argument("--split", type=str, default="train.txt")

    # model
    parser.add_argument('--model_version', type=str,
                        default='PC-Depth', choices=['SC-Depth', 'PC-Depth'])
    parser.add_argument('--resnet_layers', type=int, default=18)
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint path to load')

    # loss for sc_v1 & PC-Depth
    parser.add_argument('--photo_weight', type=float,
                        default=1.0, help='photometric loss weight')
    parser.add_argument('--geometry_weight', type=float,
                        default=0.1, help='geometry loss weight')
    parser.add_argument('--smooth_weight', type=float,
                        default=0.01, help='smoothness loss weight')

    # loss for PC-Depth
    parser.add_argument('--highlight_weight', type=float,
                        default=0.01, help='highlight loss weight')
    
    # parameters for light align
    parser.add_argument('--light_mu', type=float,
                        default=3.069096, help='mu in light align')
    parser.add_argument('--light_gamma', type=float,
                        default=2.2, help='camera gamma')

    # for ablation study
    parser.add_argument('--no_ssim', action='store_true',
                        help='use ssim in photometric loss')
    parser.add_argument('--no_auto_mask', action='store_true',
                        help='masking invalid static points')
    parser.add_argument('--no_dynamic_mask',
                        action='store_true', help='masking dynamic regions')
    parser.add_argument('--no_min_optimize', action='store_true',
                        help='optimize the minimum loss')
    parser.add_argument('--no_specular_mask', action='store_false',
                        help='no specular loss')

    
    # ablation in HS-SfMLearner
    parser.add_argument('--do_color_aug', action='store_true',
                        help='do color aug')
    
    parser.add_argument('--no_highlight_loss', action='store_true',
                        help='no highlight loss')

    parser.add_argument('--no_inpaint_smooth', action='store_true',
                        help='no inpaint in smooth loss')
    
    parser.add_argument('--no_highlight_mask', action='store_true',
                        help='no highlight mask in LP')
    
    parser.add_argument('--no_light_align', action='store_true',
                        help='not do light align')

    # training options
    parser.add_argument('--exp_name', type=str, help='experiment name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--scheduler_step_size', type=int, default=10, help='step size of the scheduler')


    # inference options
    parser.add_argument('--input_dir', type=str, help='input image path')
    parser.add_argument('--output_dir', type=str, help='output depth path')
    parser.add_argument('--save-vis', action='store_true',
                        help='save depth visualization')
    parser.add_argument('--save-depth', action='store_true',
                        help='save depth with factor 1000')

    return parser.parse_args()


def get_training_size(dataset_name):

    if dataset_name == 'c3vd':
        training_size = [256, 320]
    elif dataset_name == 'SCARED':
        training_size = [256, 320]
    else:
        print('unknown dataset type')

    return training_size
