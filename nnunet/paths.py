#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import yaml
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
import torch
from FML_backup.utils import create_config, write_value_in_config, get_config_file_path, read_config
# do not modify these unless you know what you are doing
my_output_identifier = "nnUNet"
default_plans_identifier = "nnUNetPlansv2.1"
default_data_identifier = 'nnUNetData_plans_v2.1'
default_trainer = "nnUNetTrainerV2"
default_cascade_trainer = "nnUNetTrainerV2CascadeFullRes"

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

path_to_config_file = get_config_file_path("fed_fets2021", False)
print(path_to_config_file)

words = path_to_config_file.split(os.sep)
# print(words)
dir_string = '/'
for word in words:
    if word == 'fed_kits19':
        dir_string = os.path.join(dir_string, word)
        dir_string = os.path.join(dir_string, 'dataset_creation_scripts/')
        dir_string = os.path.join(dir_string, 'dataset_location.yaml')
        break
    if word == 'fed_fets2021':
        dir_string = os.path.join(dir_string, word)
        dir_string = os.path.join(dir_string, 'dataset_creation_scripts/')
        dir_string = os.path.join(dir_string, 'dataset_location.yaml')
        break
    else:
        dir_string = os.path.join(dir_string, word)

print(dir_string)
# dict = read_config(dir_string)
if not (os.path.exists(dir_string)):
    raise FileNotFoundError("Could not find the config to read.")
with open(dir_string, "r") as file:
    dict = yaml.load(file, Loader=yaml.FullLoader)
base = dict["dataset_path"] + '/'
# with open("../../../dataset_creation_scripts/data_location.yaml", 'r') as stream:
#     base = yaml.safe_load(stream)

""" Setting the Paths based on Yaml File """

preprocessing_output_dir = base + 'fed_fets2021_preprocessing'
network_training_output_dir_base = base + 'fed_fets2021_Results'
print(base)

#
# # base = os.environ['nnUNet_raw_data_base'] if "nnUNet_raw_data_base" in os.environ.keys() else None
# # preprocessing_output_dir = os.environ['nnUNet_preprocessed'] if "nnUNet_preprocessed" in os.environ.keys() else None
# # network_training_output_dir_base = os.path.join(os.environ['RESULTS_FOLDER']) if "RESULTS_FOLDER" in os.environ.keys() else None
# myhost = os.uname()[1]
# print(myhost)
#
# base = '/home/emushtaq/kits19_data/kits19/'
# preprocessing_output_dir = '/home/emushtaq/kits19_data/kits19/preprocessing'
# fl_preprocessing_output_dir = '/home/emushtaq/kits19_data/kits19/fl_preprocessing'
# main_fl_preprocessing_output_dir = '/home/emushtaq/kits19_data/kits19/main_fl_preprocessing'
# network_training_output_dir_base = '/home/emushtaq/kits19_data/kits19/Results'
# if not torch.cuda.is_available():
#     if myhost == 'Erums-MacBook-Pro.local':
#         base = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/'
#         preprocessing_output_dir = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/preprocessing'
#         fl_preprocessing_output_dir = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/fl_preprocessing'
#         main_fl_preprocessing_output_dir = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/main_fl_preprocessing'
#         network_training_output_dir_base = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/Results'
#     if myhost == 'usc-guestwireless-upc-new119.usc.edu' or myhost == 'Erums-iMac.local' or myhost == 'usc-securewireless-student-new116.usc.edu':
#         base = '/Users/erummushtaq/Desktop/Medical_Datasets/KiTS19/kits19'
#         preprocessing_output_dir = '/Users/erummushtaq/Desktop/Medical_Datasets/KiTS19/kits19/preprocessing'
#         fl_preprocessing_output_dir = '/Users/erummushtaq/Desktop/Medical_Datasets/KiTS19/kits19/fl_preprocessing'
#         network_training_output_dir_base = '/Users/erummushtaq/Desktop/Medical_Datasets/KiTS19/kits19/Results'
#         main_fl_preprocessing_output_dir = '/Users/erummushtaq/Desktop/Medical_Datasets/KiTS19/kits19/main_fl_preprocessing'
# else:
#     # find host names for servers
#     if myhost == 'lambda-server1':
#         base = '/home/emushtaq/kits19_data/kits19/'
#         preprocessing_output_dir = '/home/emushtaq/kits19_data/kits19/preprocessing'
#         fl_preprocessing_output_dir = '/home/emushtaq/kits19_data/kits19/fl_preprocessing'
#         main_fl_preprocessing_output_dir = '/home/emushtaq/kits19_data/kits19/main_fl_preprocessing'
#         network_training_output_dir_base = '/home/emushtaq/kits19_data/kits19/Results'
#     if myhost == 'lambda-server5':
#         base = '/home/erum/Medical_datasets/kits19/'
#         preprocessing_output_dir = '/home/erum/Medical_datasets/kits19/preprocessing'
#         main_fl_preprocessing_output_dir = '/home/erum/Medical_datasets/kits19/main_fl_preprocessing'
#         fl_preprocessing_output_dir = '/home/erum/Medical_datasets/kits19/fl_preprocessing'
#         network_training_output_dir_base = '/home/erum/Medical_datasets/kits19/Results'
#     if myhost == 'lambda-server-vital':
#         base = '/home/emushtaq/Medical_datasets/kits19/'
#         preprocessing_output_dir = '/home/emushtaq/Medical_datasets/kits19/preprocessing'
#         main_fl_preprocessing_output_dir = '/home/emushtaq/Medical_datasets/kits19/main_fl_preprocessing'
#         fl_preprocessing_output_dir = '/home/emushtaq/Medical_datasets/kits19/fl_preprocessing'
#         network_training_output_dir_base = '/home/emushtaq/Medical_datasets/kits19/Results'

    # base = '../../../kits19_data/kits19/'
    # preprocessing_output_dir = '../../../kits19_data/kits19/preprocessing'
    # network_training_output_dir_base = '../../../kits19_data/kits19/Results'
# base = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/'
# preprocessing_output_dir = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/preprocessing'
# network_training_output_dir_base = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/Results'




fl_preprocessing_output_dir = main_fl_preprocessing_output_dir = preprocessing_output_dir

# print(base)
if base is not None:
    nnUNet_raw_data = join(base, "nnUNet_raw_data")
    nnUNet_cropped_data = join(base, "nnUNet_cropped_data")
    fl_nnUNet_cropped_data = join(base, "fl_nnUNet_cropped_data")
    maybe_mkdir_p(nnUNet_raw_data)
    maybe_mkdir_p(nnUNet_cropped_data)
    maybe_mkdir_p(fl_nnUNet_cropped_data)
else:
    print("nnUNet_raw_data_base is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up properly.")
    nnUNet_cropped_data = nnUNet_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
    maybe_mkdir_p(fl_preprocessing_output_dir)
    maybe_mkdir_p(main_fl_preprocessing_output_dir)
else:
    print("nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information on how to set this "
          "up.")
    network_training_output_dir = None
