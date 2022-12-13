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

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from batchgenerators.utilities.file_and_folder_operations import *
# from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import crop
from nnunet.paths import *
import shutil
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.preprocessing.sanity_checks import verify_dataset_integrity
from nnunet.training.model_restore import recursive_find_python_class
import logging
from collections import defaultdict
import csv
import numpy as np
from itertools import groupby
import nnunet

def read_csv_file(csv_file = '../../data_preprocessing/anony_sites.csv'):
    logging.info(' Read CSV File')
    # with open(csv_file, 'rb') as csvfile:
    columns = defaultdict(list) # each value in each column is appended to a list

    with open(csv_file) as f:
        reader = csv.DictReader(f) # read rows into a dictionary format
        for row in reader: # read a row as {column1: value1, column2: value2,...}
            for (k,v) in row.items(): # go over each column name and value
                if k == 'case':
                    columns[k].append(v) # append the value into the appropriate list
                                     # based on column name k
                else:
                    columns[k].append(int(v))

    logging.info('Column 0')
    case_ids = columns['case']
    logging.info(case_ids[0])
    logging.info('Column 1')
    site_ids = columns['site_id']
    logging.info(site_ids[0])

    case_ids_array = np.array(site_ids)
    unique_hospital_IDs = np.unique(case_ids_array)
    freq = {key: len(list(group)) for key, group in groupby(np.sort(case_ids_array))}

    logging.info(freq)
    logging.info(unique_hospital_IDs)
    logging.info('Total unqiue IDs '+str(len(unique_hospital_IDs)))
    logging.info('max unqiue IDs ' + str(max(unique_hospital_IDs)))
    return case_ids, site_ids[0:209], unique_hospital_IDs


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_ids", nargs="+", help="List of integers belonging to the task ids you wish to run"
                                                            " experiment planning and preprocessing for. Each of these "
                                                            "ids must, have a matching folder 'TaskXXX_' in the raw "
                                                            "data folder")
    parser.add_argument("-pl3d", "--planner3d", type=str, default="ExperimentPlanner3D_v21",
                        help="Name of the ExperimentPlanner class for the full resolution 3D U-Net and U-Net cascade. "
                             "Default is ExperimentPlanner3D_v21. Can be 'None', in which case these U-Nets will not be "
                             "configured")
    parser.add_argument("-pl2d", "--planner2d", type=str, default="ExperimentPlanner2D_v21",
                        help="Name of the ExperimentPlanner class for the 2D U-Net. Default is ExperimentPlanner2D_v21. "
                             "Can be 'None', in which case this U-Net will not be configured")
    parser.add_argument("-no_pp", action="store_true",
                        help="Set this flag if you dont want to run the preprocessing. If this is set then this script "
                             "will only run the experiment planning and create the plans file")
    parser.add_argument("-tl", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the low resolution data for the 3D low "
                             "resolution U-Net. This can be larger than -tf. Don't overdo it or you will run out of "
                             "RAM")
    parser.add_argument("-tf", type=int, required=False, default=8,
                        help="Number of processes used for preprocessing the full resolution data of the 2D U-Net and "
                             "3D U-Net. Don't overdo it or you will run out of RAM")
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument("-overwrite_plans", type=str, default=None, required=False,
                        help="Use this to specify a plans file that should be used instead of whatever nnU-Net would "
                             "configure automatically. This will overwrite everything: intensity normalization, "
                             "network architecture, target spacing etc. Using this is useful for using pretrained "
                             "model weights as this will guarantee that the network architecture on the target "
                             "dataset is the same as on the source dataset and the weights can therefore be transferred.\n"
                             "Pro tip: If you want to pretrain on Hepaticvessel and apply the result to LiTS then use "
                             "the LiTS plans to run the preprocessing of the HepaticVessel task.\n"
                             "Make sure to only use plans files that were "
                             "generated with the same number of modalities as the target dataset (LiTS -> BCV or "
                             "LiTS -> Task008_HepaticVessel is OK. BraTS -> LiTS is not (BraTS has 4 input modalities, "
                             "LiTS has just one)). Also only do things that make sense. This functionality is beta with"
                             "no support given.\n"
                             "Note that this will first print the old plans (which are going to be overwritten) and "
                             "then the new ones (provided that -no_pp was NOT set).")
    parser.add_argument("-overwrite_plans_identifier", type=str, default=None, required=False,
                        help="If you set overwrite_plans you need to provide a unique identifier so that nnUNet knows "
                             "where to look for the correct plans and data. Assume your identifier is called "
                             "IDENTIFIER, the correct training command would be:\n"
                             "'nnUNet_train CONFIG TRAINER TASKID FOLD -p nnUNetPlans_pretrained_IDENTIFIER "
                             "-pretrained_weights FILENAME'")

    args = parser.parse_args()
    task_ids = args.task_ids
    dont_run_preprocessing = args.no_pp
    tl = args.tl
    tf = args.tf
    planner_name3d = args.planner3d
    planner_name2d = args.planner2d

    if planner_name3d == "None":
        planner_name3d = None
    if planner_name2d == "None":
        planner_name2d = None

    if args.overwrite_plans is not None:
        if planner_name2d is not None:
            print("Overwriting plans only works for the 3d planner. I am setting '--planner2d' to None. This will "
                  "skip 2d planning and preprocessing.")
        assert planner_name3d == 'ExperimentPlanner3D_v21_Pretrained', "When using --overwrite_plans you need to use " \
                                                                       "'-pl3d ExperimentPlanner3D_v21_Pretrained'"

    # we need raw data
    tasks = []
    for i in task_ids:
        i = int(i)

        # task_name = convert_id_to_task_name(i) #uncoment if you have raw and cropped data
        task_name = 'Task064_KiTS_labelsFixed'
        # nnUNet_raw_data = '/Users/erummushtaq/Kidney_Datasets/KITS-21-Challenge/kits19-master/'
        if args.verify_dataset_integrity:
            verify_dataset_integrity(join(nnUNet_raw_data, task_name))

        # crop(task_name, False, tf)
        # crop(task_name, True, tf)

        tasks.append(task_name)

    search_in = join(nnunet.__path__[0], "experiment_planning")

    if planner_name3d is not None:
        planner_3d = recursive_find_python_class([search_in], planner_name3d, current_module="nnunet.experiment_planning")
        if planner_3d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name3d)
    else:
        planner_3d = None

    if planner_name2d is not None:
        planner_2d = recursive_find_python_class([search_in], planner_name2d, current_module="nnunet.experiment_planning")
        if planner_2d is None:
            raise RuntimeError("Could not find the Planner class %s. Make sure it is located somewhere in "
                               "nnunet.experiment_planning" % planner_name2d)
    else:
        planner_2d = None

    print(' Planner 2D ')
    print(planner_2d)

    print(' Planner 3D ')
    print(planner_3d)
    for t in tasks:
        print("\n\n\n", t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        cropped_out_dir = os.path.join(nnUNet_cropped_data, t)
        # maybe_mkdir_p(nnUNet_cropped_data)
        cropped_out_dir_seg = os.path.join(cropped_out_dir, 'gt_segmentations')

        # maybe_mkdir_p(nnUNet_cropped_data_new)

        hospital_ID = 0
        case_ids, site_ids, hospital_ids = read_csv_file()
        main_fl_preprocessing_output_dirr = os.path.join(main_fl_preprocessing_output_dir, t)
        maybe_mkdir_p(main_fl_preprocessing_output_dirr)
        maybe_mkdir_p(main_fl_preprocessing_output_dirr + '/nnUNetData_plans_v2.1_stage0/')
        maybe_mkdir_p(main_fl_preprocessing_output_dirr + '/gt_segmentations')
        for ID in range(hospital_ID, 89):
            # if ID == 10:
            #     exit()
            cropped_out_dir_new_ci = os.path.join(fl_nnUNet_cropped_data, str(ID))
            maybe_mkdir_p(cropped_out_dir_new_ci)
            cropped_out_dir_new_im = os.path.join(cropped_out_dir_new_ci, t)
            maybe_mkdir_p(cropped_out_dir_new_im)
            cropped_out_dir_new_seg = os.path.join(cropped_out_dir_new_im, 'gt_segmentations')
            maybe_mkdir_p(cropped_out_dir_new_seg)

            preprocessing_output_dir_new = os.path.join(fl_preprocessing_output_dir, str(ID))
            if not os.path.exists(cropped_out_dir_new_im + '/dataset.json'):
                shutil.copy(cropped_out_dir + '/dataset.json', cropped_out_dir_new_im)

            hospital_ID += 1
            client_ids = np.where(np.array(site_ids) == ID)[0]
            client_data_ids = np.array([case_ids[i] for i in client_ids])
            if client_ids.size != 0:
                if client_ids.size > 0:
                    for i in client_ids:
                        g = case_ids[i]
                        if not os.path.exists(cropped_out_dir_new_seg + '/' + g + '.nii.gz'):
                            # Move .nii.gz for gt seg
                            shutil.copy(cropped_out_dir_seg + '/' + g + '.nii.gz', cropped_out_dir_new_seg)
                            # Move .pkl and .npz
                            shutil.copy(cropped_out_dir + '/' + g + '.pkl', cropped_out_dir_new_im)
                            shutil.copy(cropped_out_dir + '/' + g + '.npz', cropped_out_dir_new_im)
                            # shutil.move(cropped_out_dir + '/' + g + '.npz', cropped_out_dir_new_im)


                    preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir_new, t)
                    # shutil.move(cropped_out_dir + g, nnUNet_cropped_data_new)
                    dataset_json = load_json(join(cropped_out_dir_new_im, 'dataset.json'))
                    modalities = list(dataset_json["modality"].values())
                    collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
                    dataset_analyzer = DatasetAnalyzer(cropped_out_dir_new_im, overwrite=False,
                                                       num_processes=tf)  # this class creates the fingerprint
                    _ = dataset_analyzer.analyze_dataset(
                        collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner

                    maybe_mkdir_p(preprocessing_output_dir_this_task)
                    shutil.copy(join(cropped_out_dir_new_im, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
                    shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)

                    threads = (tl, tf)

                    print("number of threads: ", threads, "\n")

                    if planner_3d is not None:
                        if args.overwrite_plans is not None:
                            assert args.overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
                            exp_planner = planner_3d(cropped_out_dir_new_im, preprocessing_output_dir_this_task,
                                                     args.overwrite_plans,
                                                     args.overwrite_plans_identifier)
                        else:
                            exp_planner = planner_3d(cropped_out_dir_new_im, preprocessing_output_dir_this_task)
                        exp_planner.plan_experiment()
                        if not dont_run_preprocessing:  # double negative, yooo
                            exp_planner.run_preprocessing(threads)
                    if planner_2d is not None:
                        exp_planner = planner_2d(cropped_out_dir_new_im, preprocessing_output_dir_this_task)
                        exp_planner.plan_experiment()
                        if not dont_run_preprocessing:  # double negative, yooo
                            exp_planner.run_preprocessing(threads)

                    # now move these files in a separate folder
                    #thee main folders, we use nnUNetData_plans_v2.1_stage0

                    for i in client_ids:
                        g = case_ids[i]
                        # Move .pkl and .npz of data
                        shutil.copy(preprocessing_output_dir_this_task + '/nnUNetData_plans_v2.1_stage0/' + g + '.npz', main_fl_preprocessing_output_dirr + '/nnUNetData_plans_v2.1_stage0/')
                        shutil.copy(preprocessing_output_dir_this_task + '/nnUNetData_plans_v2.1_stage0/' + g + '.pkl', main_fl_preprocessing_output_dirr + '/nnUNetData_plans_v2.1_stage0/')
                        # Also Move gt segmentations
                        shutil.copy(preprocessing_output_dir_this_task + '/gt_segmentations/' + g + '.nii.gz',
                            main_fl_preprocessing_output_dirr + '/gt_segmentations')

                # # Now delete this clients' files
                # for i in client_ids:
                #     g = case_ids[i]
                #     os.remove(cropped_out_dir_new_seg + '/' + g + '.nii.gz')
                #     os.remove(cropped_out_dir_new_im + '/' + g + '.pkl')
                #     os.remove(cropped_out_dir_new_im + '/' + g + '.npz')

            # exit()



        # cropped_out_dir_seg = cropped_out_dir + '/Task064_KiTS_labelsFixed/gt_segmentations'
        #
        # cropped_out_dir_image = cropped_out_dir + '/Task064_KiTS_labelsFixed'
        # cropped_out_dir_seg = cropped_out_dir + '/Task064_KiTS_labelsFixed/gt_segmentations'


        # print(get_files)
        # for g in get_files:
        #     filename = os.path.splitext(g)
        #     print(filename[0])
        #     exit()
        #     #if condition
        #     #move image and move segmentation (move both .npz and pkl)
        #     shutil.move(cropped_out_dir + '/' + g, cropped_out_dir_new_im)
        #     shutil.move(cropped_out_dir_seg + '/' + g, cropped_out_dir_new_seg)
        #
        #     shutil.move(cropped_out_dir + g, nnUNet_cropped_data_new)
        #
        # exit()

        # Move data to their respective folders cropped/client_ID and then delete later.
        # change the name of cropped_out_dir for each client so that only that data gets processed.
        # Everything will be saved to preprocessing file anyway. (but the plan file may not contain information for all the clients)
        # Use the old plans file?
        # dataset_properties file will be an issue. (create a function to read those files).
        # may be use the condition, if already exist then do not do copy paste dataset_properties and dataset.json file.


        #
        # preprocessing_output_dir_this_task = os.path.join(preprocessing_output_dir, t)
        # #splitted_4d_output_dir_task = os.path.join(nnUNet_raw_data, t)
        # #lists, modalities = create_lists_from_splitted_dataset(splitted_4d_output_dir_task)
        #
        # # we need to figure out if we need the intensity propoerties. We collect them only if one of the modalities is CT
        # dataset_json = load_json(join(cropped_out_dir, 'dataset.json'))
        # modalities = list(dataset_json["modality"].values())
        # collect_intensityproperties = True if (("CT" in modalities) or ("ct" in modalities)) else False
        # dataset_analyzer = DatasetAnalyzer(cropped_out_dir, overwrite=False, num_processes=tf)  # this class creates the fingerprint
        # _ = dataset_analyzer.analyze_dataset(collect_intensityproperties)  # this will write output files that will be used by the ExperimentPlanner
        #
        #
        # maybe_mkdir_p(preprocessing_output_dir_this_task)
        # shutil.copy(join(cropped_out_dir, "dataset_properties.pkl"), preprocessing_output_dir_this_task)
        # shutil.copy(join(nnUNet_raw_data, t, "dataset.json"), preprocessing_output_dir_this_task)
        #
        # threads = (tl, tf)
        #
        # print("number of threads: ", threads, "\n")
        #
        # if planner_3d is not None:
        #     if args.overwrite_plans is not None:
        #         assert args.overwrite_plans_identifier is not None, "You need to specify -overwrite_plans_identifier"
        #         exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task, args.overwrite_plans,
        #                                  args.overwrite_plans_identifier)
        #     else:
        #         exp_planner = planner_3d(cropped_out_dir, preprocessing_output_dir_this_task)
        #     exp_planner.plan_experiment()
        #     if not dont_run_preprocessing:  # double negative, yooo
        #         exp_planner.run_preprocessing(threads)
        # if planner_2d is not None:
        #     exp_planner = planner_2d(cropped_out_dir, preprocessing_output_dir_this_task)
        #     exp_planner.plan_experiment()
        #     if not dont_run_preprocessing:  # double negative, yooo
        #         exp_planner.run_preprocessing(threads)


if __name__ == "__main__":
    main()

