import argparse
import logging
import os
import random
import socket
import sys
from torch import nn
import numpy as np
import psutil
import setproctitle
import torch
import dill
import wandb
import shutil
from collections import OrderedDict
import nibabel as nib

from tqdm import tqdm

import argparse
import logging
import os
import random
import socket
import sys
import pandas as pd
import numpy as np
import psutil
import setproctitle
import torch
import wandb
import shutil
from collections import OrderedDict
import nibabel as nib

# from utils.resnet import get_resnet

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

from nnunet.training.network_training.nnUNetTrainer import nnUNetTrainer
from fed_kits19.dataset_creation_scripts.paths import *
from fed_kits19.metric import metric
from FedML.fedml_api.distributed.fedkits.FedAvgAPI import FedML_init
from nnunet.inference.predict_adapted import predict_cases, predict_from_folder

plans_file = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetPlansv2.1_plans_3D.pkl'
# plans_file = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/preprocessing/Task064_KiTS_labelsFixed/nnUNetPlansv2.1_plans_3D.pkl'

output_directory = network_training_output_dir_base
def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument("--model", type=str, default="nnunet", metavar="N", help="neural network used in training")

    parser.add_argument("--dataset", type=str, default="kits19", metavar="N", help="dataset used for training")

    parser.add_argument("--data_dir", type=str, default="./../data/", help="data directory")

    parser.add_argument("--main_round", type=int, default=100, help="max value total number of rounds - 1")


    parser.add_argument("--is_debug", type=int, default=1, help="0 or 1")

    parser.add_argument("--starting_gpu", type=int, default=0)

    parser.add_argument("--cent_preprocess", type=int, default=0)

    parser.add_argument('--partition_method', type=str, default='equal',
                        help='partition method: equal or natural')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    # Learning related
    parser.add_argument('--train_batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--eval_batch_size', type=int, default=2, metavar='N',
                        help='input batch size for evaluation (default: 8)')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for evaluation (default: 8)')
    parser.add_argument('--starting_gpu_id', type=int, default=0, metavar='N',
                        help='Starting GPU ID (default: 0)')

    # IO related
    parser.add_argument('--output_dir', type=str, default="/tmp/", metavar='N',
                        help='path to save the trained results and ckpts')

    # Federated Learning related
    parser.add_argument('--fl_algorithm', type=str, default="FedAvg",
                        help='Algorithm list: FedAvg; FedOPT; FedProx ')

    parser.add_argument('--backend', type=str, default="MPI",
                        help='Backend for Server and Client')

    parser.add_argument('--split_type', type=str, default="natural",
                        help='natural or centralized')

    parser.add_argument('--threshold', type=int, default=5,
                        help='natural or centralized')

    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we shoud use')

    parser.add_argument('--client_num_in_total', type=int, default=2, metavar='NN',
                        help='number of clients in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int,
                        default=2, metavar='NN', help='number of workers')

    parser.add_argument('--client_number', type=int,
                        default=2, metavar='NN', help='number of workers')

    parser.add_argument('--num_batches_per_epoch', type=int,
                        default=100, metavar='NN', help='number of workers')

    parser.add_argument('--epochs', type=int, default=1, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--valid_length', type=int, default=50, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--phase', type=str, default='infer', metavar='EP',
                        help='how many epochs will be trained locally')


    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate on the client (default: 0.001)')

    parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                        help='L2 penalty')

    parser.add_argument('--server_optimizer', type=str, default='sgd',
                        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--server_lr', type=float, default=0.1,
                        help='server learning rate (default: 0.001)')

    parser.add_argument('--server_momentum', type=float, default=0,
                        help='server momentum (default: 0)')

    parser.add_argument('--is_pre_trained', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument('--run_id', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    parser.add_argument(
        '--evaluate_during_training_steps', type=int, default=100, metavar='EP',
        help='the frequency of the evaluation during training')

    parser.add_argument('--frequency_of_the_test', type=int, default=10,
                        help='the frequency of the algorithms')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')

    parser.add_argument("--gpu_server_num", type=int, default=1, help="gpu_server_num")
    parser.add_argument("--local_epoch", type=int, default=1, help="gpu_server_num")

    parser.add_argument("--gpu_num_per_server", type=int, default=4, help="gpu_num_per_server")

    parser.add_argument('--is_mobile', type=int, default=0,
                        help='whether the program is running on the FedML-Mobile server side')

    # Communication related
    parser.add_argument(
        "--grpc_ipconfig_path",
        type=str,
        default="grpc_ipconfig.csv",
        help="config table containing ipv4 address of grpc server",
    )

    parser.add_argument(
        "--trpc_master_config_path",
        type=str,
        default="trpc_master_config.csv",
        help="config indicating ip address and port of the master (rank 0) node",
    )
    args = parser.parse_args()
    return args




def init_training_device(process_ID, fl_worker_num, gpu_num_per_machine):
    # initialize the mapping from process ID to GPU ID: <process ID, GPU ID>
    # logging.info(fl_worker_num)
    # logging.info(gpu_num_per_machine)
    if process_ID == 0:
        device = torch.device("cuda:"+str(args.starting_gpu) if torch.cuda.is_available() else "cpu")
        return device
    else:
    #  process_gpu_dict = dict()
    # for client_index in range(fl_worker_num):
    #     gpu_index = client_index % args.gpu_num_per_server + (args.starting_gpu + 1)
    #     # gpu_index = (client_index % gpu_num_per_machine)
    #     process_gpu_dict[client_index] = gpu_index

    # logging.info(process_gpu_dict)

        client_index = process_ID - 1
        gpu_index = client_index % args.gpu_num_per_server + (args.starting_gpu + 1)
        device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
        # logging.info(device)
    return device


if __name__ == "__main__":
    # # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    # if sys.platform == "darwin":
    #     os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.basicConfig(level=logging.INFO)
    # logging.info(args)

    # customize the process name
    str_process_name = "FedKits-:" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     format=str(process_id) + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
    #     datefmt="%a, %d %b %Y %H:%M:%S",
    # )
    hostname = socket.gethostname()
    # logging.info("#############process ID = "+ str(process_id)+ ", host name = "+ hostname
    # + "########"+ ", process ID = "+ str(os.getpid())+ ", process Name = " + str(psutil.Process(os.getpid())))

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb.init(project="FML", name="Validation "
            + "RUN_ID_" + str(args.run_id) + "-e" + str(args.epochs) + "-lr" + str(args.lr), config=args)

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    seed = 0
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    dataset_directory = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetData_plans_v2.1_stage0'

    # Device Assignment
    # Therefore, we can see that workers are assigned according to the order of machine list.
    # logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)

    print('validation')
    device = init_training_device(process_id, worker_number - 1, args.gpu_num_per_server)
    Trainer = nnUNetTrainer(plans_file, fold = 2, train_data= None, valid_data=None,
                                output_folder=output_directory, dataset_directory=dataset_directory, batch_dice=True,
                                stage=0, unpack_data=True, deterministic=True, fp16=False, device= device)
    logging.info(' Trainer initialized ')
    model = Trainer.initialize_network()

    # Load Trained model
    # path = base + 'Models/' + 'FL_' + '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(
    #     args.comm_round) + '_epoch_' + str(args.epochs) + 'round_best_model.model'
    # path = base + 'Semi_Models/' + 'Semi_FL_Preprocess_student_training' + '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(
    #     args.comm_round) + '_epoch_' + str(args.epochs) + 'round_best_model.model'
    # path = base + 'Semi_Models/' + 'Semi_FL_Preprocess_teacher_training' + '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(
    #     args.comm_round) + '_epoch_' + str(args.epochs) + 'round_best_model.model'
    # path = base + 'Models/' + 'CL_'+ '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(args.epochs) + '_local_epoch_' + str(args.local_epoch) + '_Threshold_' + str(args.threshold) +'round_best_model.model'
    
    # path = base + 'Best_Models/Run_ID_' + str(args.run_id) + '/Semi_FL_Preprocess_student_training' + '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(
    #     args.comm_round) + '_epoch_' + str(args.epochs) + '_Threshold_10round_best_model.model'
    path = base + 'Best_Models/Run_ID_' + str(args.run_id) +  '/Semi_FL_Preprocess_student_training' + '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(
        args.comm_round) + '_epoch_' + str(args.epochs) + '_Threshold_10round_best_model.model'
    # path = base + '/' + 'FL_'+ '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(args.comm_round) + '_epoch_' + str(args.epochs) +'round_best_model.model'
    
    # path = base + '/' + 'FL_'+ '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(args.comm_round) + '_local_epoch_' + str(args.epochs) + '_Threshold_' + str(args.threshold) +'round_best_model.model'
    
    # path = base  + 'CL_'+ '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(args.comm_round) + '_local_epoch_' + str(args.epochs) + '_Threshold_' + str(10) +'round_best_model.model'
    # path = base  + 'FL_'+ '_RUN_ID_' + str(args.run_id) + '_lr_' + str(args.lr) + '_round_' + str(args.comm_round) + '_epoch_' + str(args.epochs) + 'round_best_model.model'

    #CL__RUN_ID_1103_lr_0.0003_round_3000_local_epoch_1_Threshold_5round_best_model.model
    logging.info(path)
    if os.path.exists(path):  # checking if there is a file with this name
        print(" Model Found ")
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
    #
    # Threshold based directory
    # df = pd.read_csv('./four_thresholded_sites.csv')
    df = pd.read_csv('./thresholded_sites.csv')
    key = "test_" + str(1000)
    df2 = df.query("train_test_split_silo == '" + key + "' ").reset_index(drop=True)
    valid_ids = df2.case_ids.tolist()
    #
    # # Make directory for threshold
    validation_folder = nnUNet_raw_data + '/Task064_KiTS_labelsFixed/imagesTr/'
    validation_folder_gt = nnUNet_raw_data + '/Task064_KiTS_labelsFixed/labelsTr/'
    input_folder = nnUNet_raw_data + '/Validation_Images'
    input_folder_gt = nnUNet_raw_data + '/Validation_gt'
    if os.path.exists(input_folder):
        shutil.rmtree(input_folder)
    if os.path.exists(input_folder_gt):
        shutil.rmtree(input_folder_gt)
    # if not os.path.exists(input_folder) and not os.path.exists(input_folder_gt):
    maybe_mkdir_p(input_folder)
    maybe_mkdir_p(input_folder_gt)
    for i in valid_ids:
        image_file = validation_folder + i + '_0000.nii.gz'
        seg_file = validation_folder_gt + i + '.nii.gz'
        if os.path.exists(image_file) and os.path.exists(seg_file):
            logging.info(' File Found ')

            if not os.path.exists(input_folder +'/'+ i + '_0000.nii.gz'):
                print('Copying file')
                shutil.copy(image_file, input_folder)
                shutil.copy(seg_file, input_folder_gt)
            else:
                print('File already exists')
        break
    #
    # # Inference
    output_folder = nnUNet_raw_data + '/inference_Run_ID_'+str(args.run_id)
    predict_from_folder(model, Trainer, plans_file, input_folder, output_folder, (0, ),
                            save_npz = False, num_threads_preprocessing = 1, num_threads_nifti_save = 1,
                            lowres_segmentations = None,
                            part_id = 0, num_parts = 1, tta = True, mixed_precision=True,
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

        tk_dice, tu_kid_dice, tu_dice = metric(pred_file, gt_file)
        print(torch.unique(pred_file))
        print(torch.unique(gt_file))
        tumor_kidney_dice.append(tk_dice)
        kidney_dice.append(tu_kid_dice)
        tumor_dice.append(tu_dice)

        iteration += 1
        print(tumor_kidney_dice)
        print(kidney_dice)
        print(tumor_dice)
        exit()

    print('Tumor + Kidney Dice (mean) '+str(np.mean(tumor_kidney_dice)) + '_ std_' +str(np.std(tumor_kidney_dice)))
    print('Kidney Dice (mean)' + str(np.mean(kidney_dice))+ ' std ' + str(np.std(kidney_dice)))
    print(' Tumor Dice (mean) ' + str(np.mean(tumor_dice)) + ' std '+ str(np.std(tumor_dice)))

    Tumor_Acc = wandb.Table(columns=["Tumor Dice "])
    Tumor_Acc.add_data(str(tumor_dice))
    wandb.log({"Tumor Dice Val ":Tumor_Acc})

    Kidney_Acc = wandb.Table(columns=["Kidney Dice "])
    Kidney_Acc.add_data(str(kidney_dice))
    wandb.log({"Kidney Dice Val ":Kidney_Acc})

    KidneyT_Acc = wandb.Table(columns=["Kidney & tumor Dice "])
    KidneyT_Acc.add_data(str(kidney_dice))
    wandb.log({"Kidney and TumorDice Val ":KidneyT_Acc})

    wandb.log({" Mean Tumor Dice": np.mean(tumor_dice), "round": 0})
    wandb.log({" STD Tumor Dice": np.std(tumor_dice), "round": 0})

    wandb.log({" Mean Kidney Dice": np.mean(kidney_dice), "round": 0})
    wandb.log({" STD Kidney Dice": np.std(kidney_dice), "round": 0})

    wandb.log({" Mean Kidney and Tumor Dice": np.mean(tumor_dice), "round": 0})
    wandb.log({" STD Kidney and Tumor Dice": np.std(tumor_dice), "round": 0})




