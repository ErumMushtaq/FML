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

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

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

path_to_config_file = get_config_file_path("fed_kits19", False)
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

preprocessing_output_dir = base + 'kits2019_preprocessing'
network_training_output_dir_base = base + 'kits2019_Results'
# base = os.environ['nnUNet_raw_data_base'] if "nnUNet_raw_data_base" in os.environ.keys() else None
# preprocessing_output_dir = os.environ['nnUNet_preprocessed'] if "nnUNet_preprocessed" in os.environ.keys() else None
# network_training_output_dir_base = os.path.join(os.environ['RESULTS_FOLDER']) if "RESULTS_FOLDER" in os.environ.keys() else None

if base is not None:
    nnUNet_raw_data = join(base, "kits2019_nnUNet_raw_data")
    nnUNet_cropped_data = join(base, "kits2019_nnUNet_cropped_data")
    maybe_mkdir_p(nnUNet_raw_data)
    maybe_mkdir_p(nnUNet_cropped_data)
else:
    print("nnUNet_raw_data_base is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up properly.")
    nnUNet_cropped_data = nnUNet_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
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
