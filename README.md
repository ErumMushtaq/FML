# What If Kidney Tumor Segmentation Challenge (KiTS19) Never Happened
 This repository hosts the code of our ICMLA paper What If Kidney Tumor Segmentation Challenge (KiTS19) Never Happened.


##KiTS19
### Dataset Description

|                   | Dataset description 
| ----------------- | -----------------------------------------------
| Description       | This is the dataset from KiTS19 Challenge.
| Dataset           | 210 CT scans with segmentation masks as Train Data and 90 CT scans with no segmentations as Test Data. Since Test data does not have ground truth segmetation masks, we canot use it for training/testing. Therefore, we will use only 210 CT scans in our exploration of this dataset. 
| Centers           | Data comes from 87 different centers. The sites information can be found in fed_kits19/dataset_creation_scripts/anony_sites.csv file. Since most the sites have small amount of data, we set a threshold of 10 on the amount of data a silo should have, and include only those silos (total 6) that meet this threshold for the Training. This leaves us with 96 patients data. The rest of the silos data is used for Validation/testing.
| Task              | Supervised Segmentation

### Getting Started
####Data Download Commands:
The commands for data download
(as given on the official kits19 git repository (https://github.com/neheller/kits19)) are as follows,

1. Clone the kits19 git repository
```bash
git clone https://github.com/neheller/kits19
```

2. Run the following commands to download the dataset. Make sure you have ~30GB space available.
```bash
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
```
These commands will populate the data folder (given in the kits19 repository) with the imaging data. 

3. Move the downloaded dataset to the data directory you want to keep it in.
4. To store the data path (path to the data folder given by KiTS19), run the following command in the directory 'fed_kits19/dataset_creation_scripts/nnunet_library/dataset_conversion',
```bash
python3 create_config.py --output_folder "data_folder_path" 
```
Note that it should not include the name of the data folder such as 'Desktop/kits19' can be an example of the "data_folder_path" given data folder resides in the kits19 directory.


#### Preprocessing
For preprocessing, we use [nnunet](https://github.com/MIC-DKFZ/nnUNet) library and [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators) packages. We exploit nnunet preprocessing pipeline
to apply intensity normalization, voxel and foreground resampling. In addition, we apply extensive transformations such as random crop, rotation, scaling, mirror etc from the batchgenerators package. 

1. To run preprocessing, first step is dataset conversion. For this step, go to the following directory from the fed_kits19 directory
```bash
cd dataset_creation_scripts/nnunet_library/dataset_conversion
```
and run the following command to prepare the data for preprocessing.
```bash
python3 Task064_KiTS_labelsFixed.py 
```
2. After data conversion, next step is to run the preprocessing which involves, data intensity normalization and voxel resampling. To run preprocessing, run the following command to go to the right directory from fed_kits19 directory
```bash
cd dataset_creation_scripts/nnunet_library/experiment_planning
```
and run the following command to preprocess the data,
```bash
python3 nnUNet_plan_and_preprocess.py -t 064
```
For the preprocessing, it can take around ~30-45 minutes. 
With this preprocessing, running the experiments can be very time efficient as it saves the preprocessing time for every experiment run.

#### Centralized Experiment:
1. Go to fed_kits19/Centralized folder and run the following command
```bash
sh centralized_kits19.sh nnunet 0 0.0003 3000 0 kits19 1105 1 train 0
```
#### FL Experriments:
1. Go to fed_kits19/FL folder and run the following command
```bash
sh fedavg_kits19.sh 2 1 2 nnunet 3000 1 0.003 kits19 3 0 train 1301 1
```

#### Semi-Supervised FL Experiments
1. Go to fed_kits19/Semi_FL folder and run the following command
```bash
sh student_teacher_FL.sh nnunet 0 2 3e-4 1 '-' 0 10 kits19 2525 1 student_training 6 7 3000 1 mixup 6
```

#### Validation of any Experiment:
Once you have performed training, go to the fed_kits19/FL and choose the hyper-parameters (you used for running the training experiments) and run the validation.sh code. 

