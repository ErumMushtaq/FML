





import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))

from batchgenerators.utilities.file_and_folder_operations import *
import pickle
# from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join
from nnunet.experiment_planning.DatasetAnalyzer import DatasetAnalyzer
from nnunet.experiment_planning.utils import crop
from nnunet.paths import *
import shutil
import gzip
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name

i = int('064')
task_name = convert_id_to_task_name(int('064'))
for ID in range(0, 10):
    preprocessing_output_dir_new = os.path.join(fl_preprocessing_output_dir, str(ID))
    Data_file_dir =  os.path.join(preprocessing_output_dir_new, task_name)
    if os.path.exists(Data_file_dir + '/nnUNetPlansv2.1_plans_3D.pkl'):
        file_name = Data_file_dir + '/nnUNetPlansv2.1_plans_3D.pkl'
        print(file_name)
        with open(file_name, 'rb') as f:
            x = pickle.load(f)
            print(x)
    # else:
    #     print(' File does not exist ')

#
#
# for ID in range(hospital_ID, 89):
#     # if ID == 10:
#     #     exit()
#     cropped_out_dir_new_ci = os.path.join(nnUNet_cropped_data_new, str(ID))
#     maybe_mkdir_p(cropped_out_dir_new_ci)
#     cropped_out_dir_new_im = os.path.join(cropped_out_dir_new_ci, t)
#     maybe_mkdir_p(cropped_out_dir_new_im)
#     cropped_out_dir_new_seg = os.path.join(cropped_out_dir_new_im, 'gt_segmentations')
#     maybe_mkdir_p(cropped_out_dir_new_seg)
#
#     preprocessing_output_dir_new = os.path.join(fl_preprocessing_output_dir, str(ID))
#     if not os.path.exists(cropped_out_dir_new_im + '/dataset.json'):
#         shutil.copy(cropped_out_dir + '/dataset.json', cropped_out_dir_new_im)
#
#     hospital_ID += 1
#     client_ids = np.where(np.array(site_ids) == ID)[0]
#     client_data_ids = np.array([case_ids[i] for i in client_ids])
#     if client_ids.size != 0:
#         for i in client_ids:
#             g = case_ids[i]