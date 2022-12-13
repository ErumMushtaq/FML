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
from torch import nn
import shutil
from collections import OrderedDict
import nibabel as nib

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from FedML.fedml_api.distributed.semi_fedkits19.FedAvgAPI import FedML_FedAvg_distributed
from FML_backup.fed_kits19.semi_dataset import SemiFedKiTS19
from FedML.fedml_api.distributed.semi_fedkits19.FedAvgAPI import FedML_init
from fed_kits19.dataset_creation_scripts.paths import *
from nnunet.network_architecture.initialization import InitWeights_He
from fed_kits19.model import Baseline
from dataset_creation_scripts.nnunet_library.data_augmentations import transformations, semi_transformations
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, get_patch_size
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
# from nnunet.paths import *
# from evaluation_metrics.kits19_eval import evaluation

# plans_file = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetPlansv2.1_plans_3D.pkl'
# output_directory = network_training_output_dir_base
# dataset_directory = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetData_plans_v2.1_stage0'

plans_file = '/Users/erummushtaq/Desktop/Medical_Datasets/KiTS19/kits19/kits19_preprocessing/Task064_KiTS_labelsFixed/nnUNetPlansv2.1_plans_3D.pkl'
output_directory = []
dataset_directory = '/Users/erummushtaq/Desktop/Medical_Datasets/KiTS19/kits19/kits19_preprocessing/Task064_KiTS_labelsFixed/nnUNetData_plans_v2.1_stage0'
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

    parser.add_argument('--starting_gpu', type=int, default=4,
                        help='start_gpu')

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

    parser.add_argument('--threshold', type=int, default=10,
                        help='CI')

    parser.add_argument('--is_pre_trained', type=int, default=1,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--ncl', type=int, default=2,
                        help='number of label clients')

    parser.add_argument('--ncu', type=int, default=4,
                        help='number of unlabel clients')

    parser.add_argument('--client_number', type=int,
                        default=2, metavar='NN', help='number of workers')

    parser.add_argument('--gpu', type=int, default=5,
                        help='gpu')

    # Adding more args
    parser.add_argument('--weight', type=str, default='basic',
                        help='basic, threshold, expo, dynamic')

    parser.add_argument('--threshold_type', type=str, default='dynamic',
                        help='constant, dynamic')
    parser.add_argument('--apply_threshold', type=str, default='True',
                        help='True, False')
    parser.add_argument('--thresholded_value', type=float, default='0.50',
                        help='True, False') 
    parser.add_argument('--max_thresholded_value', type=float, default='0.90',
                        help='True, False') 

    # Teacher Model
    parser.add_argument('--teacher_run_id', type=int, default=5, help='teacher_run_id')   
    parser.add_argument('--teacher_lr', type=float, default=3e-4, help='teacher_lr')               
    parser.add_argument('--teacher_comm_round', type=int, default=3e-4, help='teacher_comm_round')
    parser.add_argument('--teacher_epoches', type=int, default=1, help='teacher_epoches')
    # parser.add_argument("--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server")

    #Weight Scheme
    parser.add_argument('--weight_scheme', type=str, default='dr2_weight',
                        help='org_weight, th_weight, expo_weight, dr_weight')


    args = parser.parse_args()
    return args


def create_model(args, model_name, process_id, device, output_dim = None):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    model = Baseline(1, 32, 3, 5, 2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True}, nn.Dropout3d,
                         {'p': 0, 'inplace': True}, nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, False,
                         False, lambda x: x, InitWeights_He(1e-2), False, True, True)

    # model = Baseline(1, 32, 3, 5, 2, 2, nn.Conv3d, nn.InstanceNorm3d, {'eps': 1e-5, 'affine': True}, nn.Dropout3d,
    #                  {'p': 0, 'inplace': True}, nn.LeakyReLU, {'negative_slope': 1e-2, 'inplace': True}, False,
    #                  False, lambda x: x, InitWeights_He(1e-2), False, True, True)

    # path = '/Users/erummushtaq/Desktop/PyCharmProjects/FML_Federated_Medical_Learning/datasets/fed_kits19/FL/model_final_checkpoint.model'
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
    return model

def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    logging.info(fl_worker_num)
    logging.info(gpu_num_per_machine)
    if process_ID == 0:
        device = torch.device("cuda:"+str(args.starting_gpu) if torch.cuda.is_available() else "cpu")
        return device
    else:
        client_index = process_ID - 1
        gpu_index = client_index % args.gpu_num_per_server + (args.starting_gpu + 1)
        device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
        # logging.info(device)
    return device


if __name__ == "__main__":
    # parse python script input parameters
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    parser = argparse.ArgumentParser()
    args = add_args(parser)


    # worker_number = 1
    # process_id = 0
    # customize the process name
    comm, process_id, worker_number = FedML_init()
    logging.basicConfig(level=logging.INFO,
                        # logging.basicConfig(level=logging.DEBUG,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')
    # logging.basicConfig(level=logging.INFO)
    logging.info(args)
    logging.info(' Initialization ')
    logging.info(process_id)
    logging.info(worker_number)

    str_process_name = "Fedml (single):" + str(process_id)
    setproctitle.setproctitle(str_process_name)


    # customize the log format
    logging.basicConfig(level=logging.INFO,
                        # logging.basicConfig(level=logging.DEBUG,
                        format=str(
                            process_id) + ' - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S')

    hostname = socket.gethostname()
    logging.info("#############process ID = " + str(process_id) +
                 ", host name = " + hostname + "########" +
                 ", process ID = " + str(os.getpid()) +
                 ", process Name = " + str(psutil.Process(os.getpid())))



    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(project="FML", name="SemiFL"
            + "r" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr)+"_threshold"+str(args.threshold), config=args)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))

    plans_file = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetPlansv2.1_plans_3D.pkl'
    plans = load_pickle(plans_file)
    stage_plans = plans['plans_per_stage'][0]
    patch_size = np.array(stage_plans['patch_size']).astype(int)
    data_aug_params = default_3D_augmentation_params
    data_aug_params['patch_size_for_spatialtransform'] = patch_size
    basic_generator_patch_size = get_patch_size(patch_size, data_aug_params['rotation_x'],
                                                data_aug_params['rotation_y'],
                                                data_aug_params['rotation_z'],
                                                data_aug_params['scale_range'])

    pad_kwargs_data = OrderedDict()
    pad_mode = "constant"
    need_to_pad = (np.array(basic_generator_patch_size) - np.array(patch_size)).astype(int)

    weak_tr_transform, strong_tr_transform, test_transform, conversion_transform = semi_transformations(
        data_aug_params['patch_size_for_spatialtransform'],
        data_aug_params)

    ltrain_dataset = SemiFedKiTS19(1, train=True, pooled=True, labelled=1)
    ltrain_data_num = len(ltrain_dataset)
    train_data_global = torch.utils.data.DataLoader(ltrain_dataset, batch_size=2, shuffle=True, num_workers=1)

    utrain_dataset = SemiFedKiTS19(1, train=True, pooled=True, labelled=0)
    utrain_data_num = len(utrain_dataset)
    utrain_dataloader = torch.utils.data.DataLoader(utrain_dataset, batch_size=2, shuffle=True, num_workers=1)

    train_data_local_dict = {}
    test_data_local_dict = {}
    data_local_num_dict = {}
    Client_ID_dict = {}
    valid_dataset = SemiFedKiTS19(1, train=False, pooled=True, labelled=0)
    for i in range(6):
        if i <= 1:
            client_train_data = SemiFedKiTS19(i, train=True, pooled=False, labelled=1) #labelled
            Client_ID_dict[i] = 1
        else:
            client_train_data = SemiFedKiTS19(i, train=True, pooled=False, labelled=0)  # labelled
            Client_ID_dict[i] = 0
        data_local_num_dict[i] = len(client_train_data)
        train_data_local_dict[i] = torch.utils.data.DataLoader(client_train_data, batch_size=2, shuffle=True, num_workers=1)
        test_data_local_dict[i] = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=True,
                                                               num_workers=1)

    test_data_num = 210 - sum(data_local_num_dict.values())
    train_data_num = sum(data_local_num_dict.values())
    # print(data_local_num_dict)
    # print(len(valid_dataset))


    test_dataset = SemiFedKiTS19(1, train=False, pooled=True, labelled=1)
    test_data_global = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=1)

    model = create_model(args, args.model, process_id, [], output_dim=None)
    if process_id == 1 or process_id == 2:
        logging.info(" Client ID "+str(process_id))
        loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})
    else:
        logging.info(" Client ID " + str(process_id) + " label ignored loss")
        loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {}, ignore_label = 255)
    # exit()
    # device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    # if args.phase == 'teacher_training':
        # First Train the teacher model


    if args.phase == 'student_training':
        logging.info(' Loading Teacher Model ')
        # 1. Load pretrained model
        data_folder_dir = os.path.join(base, 'Best_Models')
        model_save_folder = os.path.join(data_folder_dir, 'Run_ID_'+str(args.teacher_run_id))
        path = model_save_folder + '/Semi_FL_Preprocess_teacher_training' + '_RUN_ID_' + str(args.teacher_run_id) + '_lr_' + str(
            args.teacher_lr) + '_round_' + str(args.teacher_comm_round) + '_epoch_' + str(args.teacher_epoches) + '_Threshold_' + str(
            args.threshold) + 'round_best_model.model'
        if os.path.exists(path):  # checking if there is a file with this name
            saved_model = torch.load(path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            curr_state_dict_keys = list(model.state_dict().keys())
            for k, value in saved_model['state_dict'].items():
                key = k
                if key not in curr_state_dict_keys and key.startswith('module.'):
                    key = key[7:]
                new_state_dict[key] = value
            model.load_state_dict(new_state_dict)
        else:
            logging.info(' Pre-Trained Model does not exist at path '+str(path))
            exit()



    FedML_FedAvg_distributed(
        process_id,
        worker_number,
        device,
        comm,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,  # need to check what this variable is for
        train_data_local_dict,
        test_data_local_dict,
        args,
        test_data_num,
        weak_tr_transform, strong_tr_transform, test_transform, Client_ID_dict, loss
    )
        # 2.
        # print("---------- Load Best Teacher Model ---------")
        # self.load_teacher_model()
        # single_trainer = CentralizedTrainer(dataset, model, device, args, weak_tran, str_trans, val_tr )
        # single_trainer.train(ltrain_data_global, device, args, student_model = False)

        # Load the best teacher model and save labels
        # 1. load model
        # 2. Load unlabeled data and keys
        # 3. make predictions,
        # 4. save predictions with the key name.
        # single_trainer.make_predictions(utrain_data_global, device, args, utrain_data_num)
        # single_trainer.semitrain(ltrain_data_global, utrain_data_global, device, args, student_model = True)
        #Train Student model
    # else:
    #     print('validation')
    #     Trainer = nnUNetTrainer(plans_file, fold=2, train_data=None, valid_data=None,
    #                             output_folder=output_directory, dataset_directory=dataset_directory, batch_dice=True,
    #                             stage=0, unpack_data=True, deterministic=True, fp16=False, device=device)
    #     logging.info(' Trainer initialized ')
    #     model = Trainer.initialize_network()
    #
    #     # Load Trained model
    #     path = base + 'CL_'+ '_RUN_ID_' + str(
    #         args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(
    #         args.epochs) + '_local_epoch_' + str(args.local_epoch) + '_Threshold_' + str(
    #         args.threshold) + 'round_best_model.model'
    #     if os.path.exists(path):  # checking if there is a file with this name
    #         saved_model = torch.load(path, map_location=torch.device('cpu'))
    #
    #         new_state_dict = OrderedDict()
    #         curr_state_dict_keys = list(model.state_dict().keys())
    #         # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
    #         # match. Use heuristic to make it match
    #         for k, value in saved_model['state_dict'].items():
    #             key = k
    #             if key not in curr_state_dict_keys and key.startswith('module.'):
    #                 key = key[7:]
    #             # logging.info(key)
    #             new_state_dict[key] = value
    #
    #         model.load_state_dict(new_state_dict)
    #     else:
    #         logging.info(' Pre-Trained Model does not exist ')
    #         exit()
    #
    #     # Threshold based directory
    #     case_ids, site_ids, hospital_ids = read_csv_file()
    #     valid_ids = None
    #     for ID in range(0, 89):
    #         # hospital_ID += 1
    #         client_ids = np.where(np.array(site_ids) == ID)[0]
    #         if len(client_ids) <= args.threshold and len(client_ids) > 0:  # use for validation
    #             client_data_idxx = np.array([case_ids[i] for i in client_ids])
    #             if valid_ids is None:
    #                 valid_ids = client_data_idxx
    #             else:
    #                 valid_ids = np.concatenate((valid_ids, client_data_idxx), axis=0)
    #             if args.is_debug == 1 and ID > 5:
    #                 break
    #
    #     # Make directory for threshold
    #     validation_folder = nnUNet_raw_data + '/Task064_KiTS_labelsFixed/imagesTr/'
    #     validation_folder_gt = nnUNet_raw_data + '/Task064_KiTS_labelsFixed/labelsTr/'
    #     input_folder = nnUNet_raw_data + '/CL_'+ '_RUN_ID_' + str(
    #         args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(
    #         args.epochs) + '_local_epoch_' + str(args.local_epoch) + '_Threshold_' + str(
    #         args.threshold)
    #     # input_folder_gt = nnUNet_raw_data + '/Threshold_' + str(args.threshold) + '_gt_'
    #     # if not os.path.exists(input_folder) and not os.path.exists(input_folder_gt):
    #     maybe_mkdir_p(input_folder)
    #     # maybe_mkdir_p(input_folder_gt)
    #     for i in valid_ids:
    #         image_file = validation_folder + i + '_0000.nii.gz'
    #         seg_file = validation_folder_gt + i + '.nii.gz'
    #         if os.path.exists(image_file) and os.path.exists(seg_file):
    #             logging.info(' File Found ')
    #             shutil.copy(image_file, input_folder)
    #             # shutil.copy(seg_file, input_folder_gt)
    #
    #     # Inference
    #     output_folder = nnUNet_raw_data + '/inference/threshold_' + str(args.threshold)
    #     predict_from_folder(model, Trainer, plans_file, input_folder, output_folder, (0,),
    #                         save_npz=False, num_threads_preprocessing=1, num_threads_nifti_save=1,
    #                         lowres_segmentations=None,
    #                         part_id=0, num_parts=1, tta=True, mixed_precision=True,
    #                         overwrite_existing=True, mode='normal', overwrite_all_in_gpu=None,
    #                         step_size=0.5, checkpoint_name="model_final_checkpoint",
    #                         segmentation_export_kwargs=None, disable_postprocessing=True)
    #     # Dice Coefficients
    #     iteration = 0
    #     tumor_kidney_dice = []
    #     kidney_dice = []
    #     tumor_dice = []
    #     for i in valid_ids:
    #         pred_file = torch.from_numpy(nib.load(output_folder + '/' + i + '.nii.gz').get_fdata())
    #         gt_file = torch.from_numpy(nib.load(validation_folder_gt + i + '.nii.gz').get_fdata())
    #
    #         tk_dice, tu_kid_dice, tu_dice = evaluation(pred_file, gt_file)
    #         tumor_kidney_dice.append(tk_dice)
    #         kidney_dice.append(tu_kid_dice)
    #         tumor_dice.append(tu_dice)
    #
    #         iteration += 1
    #         print(tumor_kidney_dice)
    #         print(kidney_dice)
    #         print(tumor_dice)
    #
    #     print('Tumor + Kidney Dice (mean) ' + str(np.mean(tumor_kidney_dice)) + '_ std_' + str(
    #         np.std(tumor_kidney_dice)))
    #     print('Kidney Dice (mean)' + str(np.mean(kidney_dice)) + ' std ' + str(np.std(kidney_dice)))
    #     print(' Tumor Dice (mean) ' + str(np.mean(tumor_dice)) + ' std ' + str(np.std(tumor_dice)))
