import argparse

class Arguments:
    def __init__(self):
        self._parser = argparse.ArgumentParser(description='Arguments for TreeGAN.')

        # Dataset arguments
        self._parser.add_argument('--dataset_path', type=str, default='dataset/shapenetcore_partanno_segmentation_benchmark_v0', help='Dataset file path.')
        self._parser.add_argument('--class_choice', type=str, default='Airplane', help='Select one class to generate. [Airplane, Chair, ...] (default:all_class)')
        self._parser.add_argument('--batch_size', type=int, default=32, help='Integer value for batch size.')
        self._parser.add_argument('--point_num', type=int, default=2048, help='Integer value for number of points.')

        # Training arguments
        self._parser.add_argument('--gpu', type=int, default=1, help='GPU number to use.')
        self._parser.add_argument('--epochs', type=int, default=2000, help='Integer value for epochs.')
        self._parser.add_argument('--lr', type=float, default=1e-4, help='Float value for learning rate.')
        self._parser.add_argument('--ckpt_path', type=str, default='model/ckpt/', help='Checkpoint path.')
        self._parser.add_argument('--ckpt_save', type=str, default='ckpt_', help='Checkpoint name to save.')
        self._parser.add_argument('--ckpt_load', type=str, help='Checkpoint name to load. (default:None)')
        self._parser.add_argument('--result_path', type=str, default='model/generated/', help='Generated results path.')
        self._parser.add_argument('--result_save', type=str, default='pc_epoch', help='Generated results name to save.')
        self._parser.add_argument('--visdom_port', type=int, default=8097, help='Visdom port number. (default:8097)')
        self._parser.add_argument('--visdom_color', type=int, default=4, help='Number of colors for visdom pointcloud visualization. (default:4)')

        # Network arguments
        self._parser.add_argument('--lambdaGP', type=int, default=10, help='Lambda for GP term.')
        self._parser.add_argument('--D_iter', type=int, default=5, help='Number of iterations for discriminator.')
        self._parser.add_argument('--D_FEAT', type=int, default=[3,  64,  128, 256, 512, 512], nargs='+', help='Features for discriminator.')


    def parser(self):
        return self._parser
