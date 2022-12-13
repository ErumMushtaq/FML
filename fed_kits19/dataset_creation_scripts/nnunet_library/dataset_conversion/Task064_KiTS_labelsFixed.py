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


import shutil
import argparse
from batchgenerators.utilities.file_and_folder_operations import *
import sys
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
import os
import csv
import yaml
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from dataset_creation_scripts.paths import nnUNet_raw_data, base
from FML_backup.utils import read_config, write_value_in_config, get_config_file_path





def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # parser.add_argument("--output_folder", type=str, default="True", metavar="N", required=True,
    #                     help="Specify if debug mode (True) or not (False)")
    parser.add_argument("--debug", action="store_true",  help="Specify if debug mode (True) or not (False)")
    args = parser.parse_args()
    return args

def read_csv_file(csv_file = '../../../metadata/anony_sites.csv', debug = False):
    print(' Reading kits19 Meta Data ...')
    columns = defaultdict(list)  # each value in each column is appended to a list

    with open(csv_file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        iteration = 0
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                if k == 'case':
                    columns[k].append(v) # append the value into the appropriate list
                                     # based on column name k
                else:
                    columns[k].append(int(v))
            iteration += 1
            if iteration == 210:
                break

    case_ids = columns['case']
    site_ids = columns['site_id']

    case_ids_array = np.array(site_ids)
    unique_hospital_IDs = np.unique(case_ids_array[0:210])
    # print(len(unique_hospital_IDs))
    # exit()
    print(iteration)
    freq = {key: len(list(group)) for key, group in groupby(np.sort(case_ids_array))}
    print(freq)
    print(unique_hospital_IDs)
    ax = plt.axes()
    barlist = plt.bar(freq.keys(),freq.values(),color = 'g')
    print(barlist)
    # Originnal
    # barlist[6].set_color('y')
    # barlist[11].set_color('y')
    # barlist[12].set_color('g')
    # barlist[48].set_color('g')
    # barlist[50].set_color('g')
    # barlist[59].set_color('g')

    # barlist[6].set_color('g')
    # barlist[11].set_color('g')
    # barlist[12].set_color('y')
    # barlist[48].set_color('y')
    # barlist[50].set_color('y')
    # barlist[59].set_color('y')

    # barlist[6].set_color('y')
    # barlist[12].set_color('y')
    
    # barlist[17].set_color('m')
    # barlist[4:6].set_color('g')
    plt.ylabel(' Local Training Dataset Size', fontsize = 15)
    plt.xlabel(' Silo ID', fontsize = 15)

    colors = {'Train Silo':'g', 'Test Silo':'k'}  
    # colors = {'Train Silo (S)':'g', 'Train Silo (U)':'y', 'Test Silo':'k'}  
    # colors = {'Train Silo (U)':'g', 'Train Silo (S)':'y', 'Test Silo':'k'}          
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    # plt.legend(handles, labels, fontsize = 15)
    # ax.set_xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
    plt.savefig('KiTS19_test_train_full_original.png')
    plt.savefig('KiTS19_test_train.pdf')
    exit()
    plt.figure(0)
    plt.hist(site_ids[0:209], bins=len(unique_hospital_IDs)+5, color='g')
    plt.xlabel('Silo ID')
    plt.ylabel('Local Training Dataset Size')
    
    plt.title("Min = " + str(min(freq)) + ", Max = " + str(max(freq)))
    plt.savefig('kits19_Silo_vs_Data_count.png', dpi=200)
  

    # Now apply Thresholding
    thresholded_case_ids = None
    # case_ids_array.shape

    train_case_ids = case_ids[0:210]
    train_site_ids = site_ids[0:210]

    ten_thresholded_sites = dict()
    all_sites = dict()
    all_ID = 0
    IDD = 0
    if debug == False: # Load all silos
        for ID in range(0, 89):
            client_ids = np.where(np.array(train_site_ids) == ID)[0]
            if len(client_ids) > 0:
                all_sites[all_ID] = len(client_ids)
                all_ID += 1
            # all_sites
            if len(client_ids) >= 10:
                client_data_idxx = np.array([train_case_ids[i] for i in client_ids])
                ten_thresholded_sites[IDD] = len(client_ids)
                IDD += 1
                if thresholded_case_ids is None:
                    thresholded_case_ids = client_data_idxx
                else:
                    thresholded_case_ids = np.concatenate((thresholded_case_ids, client_data_idxx), axis=0)
    else:
        silo_count = 0
        for ID in range(0, 89):
            client_ids = np.where(np.array(train_site_ids) == ID)[0]
            if len(client_ids) >= 10:
                silo_count += 1
                client_data_idxx = np.array([train_case_ids[i] for i in client_ids])
                if thresholded_case_ids is None:
                    thresholded_case_ids = client_data_idxx
                else:
                    thresholded_case_ids = np.concatenate((thresholded_case_ids, client_data_idxx), axis=0)

            if silo_count == 2:
                break

    # plt.hist(ten_thresholded_sites.values(), bins=6,)

    plt.figure(3)
    plt.bar(list(all_sites.keys()), all_sites.values(), color='g')
    plt.xlabel('Silo ID')
    plt.ylabel('Local Training Dataset Size')
    plt.title("Min = " + str(min(all_sites.values())) + ", Max = " + str(max(all_sites.values())))
    plt.savefig('kits19_Silo_vs_Data_count_thresholdall.png', dpi=200)
    print(" Creating Thresholded Data's metadata file ")
    
    plt.figure(1)
    plt.bar(list(ten_thresholded_sites.keys()), ten_thresholded_sites.values(), color='g')
    plt.xlabel('Silo ID')
    plt.ylabel('Local Training Dataset Size')
    plt.title("Min = " + str(min(ten_thresholded_sites.values())) + ", Max = " + str(max(ten_thresholded_sites.values())))
    plt.savefig('kits19_Silo_vs_Data_count_threshold10.png', dpi=200)
    print(" Creating Thresholded Data's metadata file ")
    with open('../../../metadata/2_thresholded_sites.csv', 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(['case_ids', 'site_ids', 'train_test_split','train_test_split_silo'])
        silo_count = 0
        four_threshold_sites = dict()
        for ID in range(0, 89):
            client_ids = np.where(np.array(train_site_ids) == ID)[0]
            if len(client_ids) >= 4:
                print(len(client_ids))
                four_threshold_sites[silo_count] = len(client_ids)
                client_data_idxx = np.array([train_case_ids[i] for i in client_ids])
                data_length = len(client_data_idxx)
                train_ids = int(1*data_length)
                # test_ids = int(0.2*data_length)
                # print(train_ids)
                # print(test_ids)
                # print(client_data_idxx)
                # print(client_data_idxx[:train_ids])
                # print(client_data_idxx[train_ids:])
                for i in client_data_idxx[:train_ids]:
                    writer.writerow([i,silo_count,'train', 'train_'+str(silo_count)])
                # for i in client_data_idxx[train_ids:]:
                #     writer.writerow([i,silo_count,'test', 'test_'+str(silo_count)])

                if thresholded_case_ids is None:
                    thresholded_case_ids = client_data_idxx
                else:
                    thresholded_case_ids = np.concatenate((thresholded_case_ids, client_data_idxx), axis=0)
                silo_count += 1

            else:
                client_data_idxx = np.array([train_case_ids[i] for i in client_ids])
                data_length = len(client_data_idxx)
                train_ids = int(1 * data_length)
                # test_ids = int(0.2*data_length)
                # print(train_ids)
                # print(test_ids)
                # print(client_data_idxx)
                # print(client_data_idxx[:train_ids])
                # print(client_data_idxx[train_ids:])
                for i in client_data_idxx:
                    writer.writerow([i, 1000, 'test', 'test_' + str(1000)]) #1000
                # for i in client_data_idxx:
                #     writer.writerow([i, silo_count, 'test', 'test_' + str(1000)]) #1000

                if thresholded_case_ids is None:
                    thresholded_case_ids = client_data_idxx
                else:
                    thresholded_case_ids = np.concatenate((thresholded_case_ids, client_data_idxx), axis=0)
    # plt.hist(ten_thresholded_sites.values(), bins=len(ten_thresholded_sites.keys()),)
    # plt.savefig('kits19_Silo_vs_Data_count_threshold4.png', dpi=200)
    plt.figure(2)
    plt.bar(list(four_threshold_sites.keys()), four_threshold_sites.values(), color='g')
    plt.xlabel('Silo ID')
    plt.ylabel('Local Training Dataset Size')
    plt.title("Min = " + str(min(four_threshold_sites.values())) + ", Max = " + str(max(four_threshold_sites.values())))
    plt.savefig('kits19_Silo_vs_Data_count_thresholdfour.png', dpi=200)
    print(" Creating Thresholded Data's metadata file ")
    # exit()
    print(" Creating Thresholded Data's metadata file ")
    with open('../../../metadata/semi_thresholded_sites.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['case_ids', 'site_ids', 'train_test_split', 'train_test_split_silo'])
        silo_count = 0
        for ID in range(0, 89):
            client_ids = np.where(np.array(train_site_ids) == ID)[0]
            if len(client_ids) >= 10:
                client_data_idxx = np.array([train_case_ids[i] for i in client_ids])
                data_length = len(client_data_idxx)
                train_ids = int(1 * data_length)
                # test_ids = int(0.2*data_length)
                # print(train_ids)
                # print(test_ids)
                # print(client_data_idxx)
                # print(client_data_idxx[:train_ids])
                # print(client_data_idxx[train_ids:])
                if silo_count <= 1:
                    for i in client_data_idxx[:train_ids]:
                        writer.writerow([i, silo_count, 'train_l', 'train_' + str(silo_count)])
                else:
                    for i in client_data_idxx[:train_ids]:
                        writer.writerow([i, silo_count, 'train_u', 'train_' + str(silo_count)])
                # for i in client_data_idxx[train_ids:]:
                #     writer.writerow([i,silo_count,'test', 'test_'+str(silo_count)])

                if thresholded_case_ids is None:
                    thresholded_case_ids = client_data_idxx
                else:
                    thresholded_case_ids = np.concatenate((thresholded_case_ids, client_data_idxx), axis=0)
                silo_count += 1

            else:
                client_data_idxx = np.array([train_case_ids[i] for i in client_ids])
                data_length = len(client_data_idxx)
                train_ids = int(1 * data_length)
                # test_ids = int(0.2*data_length)
                # print(train_ids)
                # print(test_ids)
                # print(client_data_idxx)
                # print(client_data_idxx[:train_ids])
                # print(client_data_idxx[train_ids:])
                for i in client_data_idxx:
                    writer.writerow([i, silo_count, 'test', 'test_' + str(1000)])  # 1000

                if thresholded_case_ids is None:
                    thresholded_case_ids = client_data_idxx
                else:
                    thresholded_case_ids = np.concatenate((thresholded_case_ids, client_data_idxx), axis=0)

    return case_ids, site_ids[0:210], unique_hospital_IDs, thresholded_case_ids.tolist()


if __name__ == "__main__":
    """
    This is the KiTS dataset after Nick fixed all the labels that had errors. Downloaded on Jan 6th 2020    
    """


    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    case_ids, site_ids, unique_hospital_ids, thresholded_ids = read_csv_file(debug=args.debug)

    # exit()
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
    # if dict["download_complete"]:
    #     print("You have already downloaded the slides, aborting.")
    #     sys.exit()
    base = base + "data"
    task_id = 64
    task_name = "KiTS_labelsFixed"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_patient_names = []
    test_patient_names = []
    all_cases = subfolders(base, join=False)


    # for i in thresholded_ids:
    #     train_patients.append(all_cases[i])
    # print(train_patients)
    print(thresholded_ids)
    if args.debug == True:
        train_patients = thresholded_ids[:25]
        test_patients = all_cases[210:211] # we do not need the test data
    else:
        train_patients = thresholded_ids
        test_patients = all_cases[210:211] # we do not need the test data

    for p in train_patients:
        curr = join(base, p)
        label_file = join(curr, "segmentation.nii.gz")
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagestr, p + "_0000.nii.gz"))
        shutil.copy(label_file, join(labelstr, p + ".nii.gz"))
        train_patient_names.append(p)

    for p in test_patients:
        curr = join(base, p)
        image_file = join(curr, "imaging.nii.gz")
        shutil.copy(image_file, join(imagests, p + "_0000.nii.gz"))
        test_patient_names.append(p)

    json_dict = {}
    json_dict['name'] = "KiTS"
    json_dict['description'] = "kidney and kidney tumor segmentation"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "KiTS data for nnunet_library"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Kidney",
        "2": "Tumor"
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                             train_patient_names]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i.split("/")[-1] for i in test_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    write_value_in_config(dir_string, "download_complete", True)

