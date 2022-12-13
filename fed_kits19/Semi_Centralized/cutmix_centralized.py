import argparse
import logging
import os
import random
import socket
import sys

import numpy as np
import psutil
import setproctitle
import torch
import wandb
import shutil
from collections import OrderedDict
import nibabel as nib

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

# from FedML.fedml_api.centralized.SemiSupervisedTrainer import CentralizedTrainer
from FedML.fedml_api.centralized.cutmixtrainer import StudentTeacherTrainer
from FedML.fedml_api.centralized.FixMatchTrainer import FixMatchTrainer
from FedML.fedml_api.centralized.ProposedTrainer import FixmixTrainer
from FedML.fedml_api.centralized.CentralizedTrainer import CentralizedTrainer
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from fed_kits19.centralized_semi_cutmix import SemiKiTS19Raw, TwoStreamBatchSampler
# from nnunet.paths import *
from nnunet.network_architecture.initialization import InitWeights_He
from fed_kits19.dataset_creation_scripts.paths import *
from fed_kits19.model import Baseline
from fed_kits19.metric import metric
from torch import nn

plans_file = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetPlansv2.1_plans_3D.pkl'
output_directory = network_training_output_dir_base
dataset_directory = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetData_plans_v2.1_stage0'
def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='mobilenet', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--phase', type=str, default='training', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    parser.add_argument('--client_num_in_total', type=int, default=1000, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--run_id', type=int, default=4, metavar='NN',
                        help='number of workers')

    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--frequency_of_train_acc_report', type=int, default=1,
                        help='the frequency of training accuracy report')

    parser.add_argument('--frequency_of_test_acc_report', type=int, default=1,
                        help='the frequency of test accuracy report')

    parser.add_argument('--gpu_server_num', type=int, default=1,
                        help='gpu_server_num')

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")

    parser.add_argument('--starting_gpu_id', type=int, default=4,
                        help='start_gpu_id')

    parser.add_argument('--is_debug', type=int, default=0, help='debug mode')

    parser.add_argument("--pretrained_dir", type=str,
                        default="./../../fedml_api/model/cv/pretrained/Transformer/vit/ViT-B_16.npz",
                        help="Where to search for pretrained vit models.")

    parser.add_argument('--gpu_num_per_server', type=int, default=4,
                        help='gpu_num_per_server')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument('--frequency_of_the_test', type=int, default=20,
                        help='CI')

    parser.add_argument('--local_epoch', type=int, default=1,
                        help='Local Epoch')

    parser.add_argument('--threshold', type=int, default=5,
                        help='CI')

    parser.add_argument('--consistency_threshold', type=int, default=0.5,
                        help='CI')

    parser.add_argument('--ncl', type=int, default=2,
                        help='number of label clients')

    parser.add_argument('--ncu', type=int, default=4,
                        help='number of unlabel clients')

    parser.add_argument('--gpu', type=int, default=5,
                        help='gpu')

    parser.add_argument('--is_pre_trained', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    # label and unlabel
    parser.add_argument('--labeled_bs', type=int, default=12,
                        help='labeled_batch_size per gpu')
    parser.add_argument('--labeled_num', type=int, default=300,
                        help='labeled data')
    parser.add_argument('--ict_alpha', type=int, default=0.2,
                        help='ict_alpha')
    # costs
    parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
    parser.add_argument('--consistency_type', type=str,
                        default="mse", help='consistency_type')
    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=200.0, help='consistency_rampup')

    parser.add_argument('--ssl_method', type=str,
                        default='mixup', help='mixup, fixmatch')    
    parser.add_argument('--nnunet_tr', type=int,
                        default=1, help='1 or 0')             

    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if dataset_name == "kits19":
        data_loader = load_semisup_centralized_kits19_threshold
        # data_loader = load_partition_data_kits19
    else:
        data_loader = None

    return data_loader


def create_model(args, model_name, ema=False):
    model = None
    model = Baseline(1, 32, 3, 5, 2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True}, nn.Dropout3d,
                     {'p': 0, 'inplace': True}, nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, False,
                     False, lambda x: x, InitWeights_He(1e-2), False, True, True)
    if args.is_pre_trained == 1:
        path = './model_final_checkpoint.model'
        saved_model = torch.load(path, map_location=torch.device('cpu'))

        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(model.state_dict().keys())
        for k, value in saved_model['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            # logging.info(key)
            new_state_dict[key] = value

        model.load_state_dict(new_state_dict)
        logging.info(' Pre-Trained Model Added')

    if ema:
        for param in model.parameters():
            param.detach_()
    return model


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

    worker_number = 1
    process_id = 0
    # customize the process name
    str_process_name = "Fedml (single):" + str(process_id)
    setproctitle.setproctitle(str_process_name)


    
    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(project="FML", name="SemiCL"
            + "r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr), config=args)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    # Load Model 
    model = create_model(args, model_name=args.model)
    ema_model = create_model(args, model_name=args.model, ema=True)

    # load data
    test_dataset = SemiKiTS19Raw(train=False, labelled=0)
    test_data_global = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=1)
    # if args.phase == "semi_training":
    logging.info(" Semi Supervised Training")
    train_dataset = SemiKiTS19Raw(train = "train", nnunet_transform = args.nnunet_tr)
    labeled_idxs = train_dataset.get_labelled_indices()
    unlabeled_idxs = train_dataset.get_unlabelled_indices()

    logging.info(str(len(labeled_idxs))+' Labeled Indices ')
    logging.info(str(len(unlabeled_idxs))+' Un-Labeled Indices ')
    if args.ssl_method == 'mixup' or args.ssl_method == 'fixmix':
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, 3, 3-1)
    elif args.ssl_method == 'fixmatch':
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs, 2, 2-1) # one labelled and one unlabelled example 

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler= batch_sampler, pin_memory=True)
    dataset = [len(train_dataloader), len(test_data_global), train_dataloader, test_data_global]
    [train_data_num, test_data_num, train_data_global, test_data_global] = dataset
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    #student_model, teacher_model, dataset, device, args, weak_tr, strong_tr, val_tr

    if args.ssl_method == 'mixup':
        Trainer = StudentTeacherTrainer(model, ema_model, dataset, device, args)
    elif args.ssl_method == 'fixmatch':            
        Trainer = FixMatchTrainer(model, ema_model, dataset, device, args)
        logging.info('Trainer called ')
    elif args.ssl_method == 'fixmix':            
        Trainer = FixmixTrainer(model, ema_model, dataset, device, args)
        logging.info('Trainer called ')
    Trainer.train()
    # else:
    #     logging.info(" Supervised Training")
    #     train_dataset = SemiKiTS19Raw( train = "train", labelled = 1)
    #     logging.info("Training data size "+str(len(train_dataset)))
    #     logging.info("Test data size "+str(len(test_dataset)))
    #     train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1)
    #     dataset = [len(train_dataloader), len(test_data_global), train_dataloader, test_data_global]
    #     device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    #     Trainer = CentralizedTrainer( dataset, model, device, args)
    #     Trainer.train(train_dataloader, device, args)

    