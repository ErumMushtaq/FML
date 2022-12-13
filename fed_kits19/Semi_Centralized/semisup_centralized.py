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

from FedML.fedml_api.centralized.SemiSupervisedTrainer import CentralizedTrainer
from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from nnunet.paths import *

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

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='CI')

    parser.add_argument('--local_epoch', type=int, default=1,
                        help='Local Epoch')

    parser.add_argument('--threshold', type=int, default=5,
                        help='CI')

    parser.add_argument('--ncl', type=int, default=2,
                        help='number of label clients')

    parser.add_argument('--ncu', type=int, default=4,
                        help='number of unlabel clients')

    parser.add_argument('--gpu', type=int, default=5,
                        help='gpu')

    args = parser.parse_args()
    return args


def load_data(args, dataset_name):
    if dataset_name == "kits19":
        data_loader = load_semisup_centralized_kits19_threshold
        # data_loader = load_partition_data_kits19
    else:
        data_loader = None

    return data_loader


def create_model(args, model_name):
    model = None
    model = Baseline(1, 32, 3, 5, 2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True}, nn.Dropout3d,
                     {'p': 0, 'inplace': True}, nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, False,
                     False, lambda x: x, InitWeights_He(1e-2), False, True, True)
    return model


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    worker_number = 1
    process_id = 0
    # customize the process name
    str_process_name = "Fedml (single):" + str(process_id)
    setproctitle.setproctitle(str_process_name)


    logging.basicConfig(level=logging.INFO)
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

    # load data
    # load data

    ltrain_dataset = SemiFedKiTS19(1, train=True, pooled=True, labelled=1)
    ltrain_data_num = len(ltrain_dataset)
    ltrain_data_global = torch.utils.data.DataLoader(ltrain_dataset, batch_size=2, shuffle=True, num_workers=1)

    utrain_dataset = SemiFedKiTS19(1, train=True, pooled=True, labelled=0)
    utrain_data_num = len(utrain_dataset)
    utrain_dataloader = torch.utils.data.DataLoader(utrain_dataset, batch_size=2, shuffle=True, num_workers=1)

    test_dataset = SemiFedKiTS19(1, train=False, pooled=True, labelled=1)
    test_data_global = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=1)


    # dataloader = load_data(args, args.dataset)
    # ltrain_data_num, utrain_data_num, test_data_num, ltrain_data_global, utrain_data_global, test_data_global, weak_tran, str_trans, val_tr = load_semisup_centralized_kits19_threshold(dataset_directory, plans_file, 0, args.ncl, args.ncu, threshold = args.threshold)
    dataset = [ltrain_data_num, test_data_num, utrain_data_global, test_data_global]

    # _ = next(utrain_data_global)
    # exit()

    model = create_model(args, model_name=args.model)
    # exit()
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    if args.phase == 'training':
        # First Train the teacher model
        single_trainer = CentralizedTrainer(dataset, model, device, args, weak_tran, str_trans, val_tr )
        single_trainer.train(ltrain_data_global, device, args, student_model = False)

        # Load the best teacher model and save labels
        # 1. load model
        # 2. Load unlabeled data and keys
        # 3. make predictions,
        # 4. save predictions with the key name.
        single_trainer.make_predictions(utrain_data_global, device, args, utrain_data_num)
        single_trainer.semitrain(ltrain_data_global, utrain_data_global, device, args, student_model = True)
        #Train Student model
    else:
        print('validation')
        Trainer = nnUNetTrainer(plans_file, fold=2, train_data=None, valid_data=None,
                                output_folder=output_directory, dataset_directory=dataset_directory, batch_dice=True,
                                stage=0, unpack_data=True, deterministic=True, fp16=False, device=device)
        logging.info(' Trainer initialized ')
        model = Trainer.initialize_network()

        # Load Trained model
        path = base + 'CL_'+ '_RUN_ID_' + str(
            args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(
            args.epochs) + '_local_epoch_' + str(args.local_epoch) + '_Threshold_' + str(
            args.threshold) + 'round_best_model.model'
        if os.path.exists(path):  # checking if there is a file with this name
            saved_model = torch.load(path, map_location=torch.device('cpu'))

            new_state_dict = OrderedDict()
            curr_state_dict_keys = list(model.state_dict().keys())
            # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
            # match. Use heuristic to make it match
            for k, value in saved_model['state_dict'].items():
                key = k
                if key not in curr_state_dict_keys and key.startswith('module.'):
                    key = key[7:]
                # logging.info(key)
                new_state_dict[key] = value

            model.load_state_dict(new_state_dict)
        else:
            logging.info(' Pre-Trained Model does not exist ')
            exit()

        # Threshold based directory
        case_ids, site_ids, hospital_ids = read_csv_file()
        valid_ids = None
        for ID in range(0, 89):
            # hospital_ID += 1
            client_ids = np.where(np.array(site_ids) == ID)[0]
            if len(client_ids) <= args.threshold and len(client_ids) > 0:  # use for validation
                client_data_idxx = np.array([case_ids[i] for i in client_ids])
                if valid_ids is None:
                    valid_ids = client_data_idxx
                else:
                    valid_ids = np.concatenate((valid_ids, client_data_idxx), axis=0)
                if args.is_debug == 1 and ID > 5:
                    break

        # Make directory for threshold
        validation_folder = nnUNet_raw_data + '/Task064_KiTS_labelsFixed/imagesTr/'
        validation_folder_gt = nnUNet_raw_data + '/Task064_KiTS_labelsFixed/labelsTr/'
        input_folder = nnUNet_raw_data + '/CL_'+ '_RUN_ID_' + str(
            args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(
            args.epochs) + '_local_epoch_' + str(args.local_epoch) + '_Threshold_' + str(
            args.threshold)
        # input_folder_gt = nnUNet_raw_data + '/Threshold_' + str(args.threshold) + '_gt_'
        # if not os.path.exists(input_folder) and not os.path.exists(input_folder_gt):
        maybe_mkdir_p(input_folder)
        # maybe_mkdir_p(input_folder_gt)
        for i in valid_ids:
            image_file = validation_folder + i + '_0000.nii.gz'
            seg_file = validation_folder_gt + i + '.nii.gz'
            if os.path.exists(image_file) and os.path.exists(seg_file):
                logging.info(' File Found ')
                shutil.copy(image_file, input_folder)
                # shutil.copy(seg_file, input_folder_gt)

        # Inference
        output_folder = nnUNet_raw_data + '/inference/threshold_' + str(args.threshold)
        predict_from_folder(model, Trainer, plans_file, input_folder, output_folder, (0,),
                            save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1,
                            lowres_segmentations=None,
                            part_id=0, num_parts=1, tta=True, mixed_precision=True,
                            overwrite_existing=True, mode='normal', overwrite_all_in_gpu=None,
                            step_size=0.5, checkpoint_name="model_final_checkpoint",
                            segmentation_export_kwargs=None, disable_postprocessing=True)
        # Dice Coefficients
        iteration = 0
        tumor_kidney_dice = []
        kidney_dice = []
        tumor_dice = []
        for i in valid_ids:
            pred_file = torch.from_numpy(nib.load(output_folder + '/' + i + '.nii.gz').get_fdata())
            gt_file = torch.from_numpy(nib.load(validation_folder_gt + i + '.nii.gz').get_fdata())

            tk_dice, tu_kid_dice, tu_dice = evaluation(pred_file, gt_file)
            tumor_kidney_dice.append(tk_dice)
            kidney_dice.append(tu_kid_dice)
            tumor_dice.append(tu_dice)

            iteration += 1
            print(tumor_kidney_dice)
            print(kidney_dice)
            print(tumor_dice)

        print('Tumor + Kidney Dice (mean) ' + str(np.mean(tumor_kidney_dice)) + '_ std_' + str(
            np.std(tumor_kidney_dice)))
        print('Kidney Dice (mean)' + str(np.mean(kidney_dice)) + ' std ' + str(np.std(kidney_dice)))
        print(' Tumor Dice (mean) ' + str(np.mean(tumor_dice)) + ' std ' + str(np.std(tumor_dice)))
