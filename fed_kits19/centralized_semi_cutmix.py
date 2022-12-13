import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nnunet
import random
import os
import sys
from torch.utils.data.sampler import Sampler
import itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
from mask_gen_file import BoxMaskGenerator
from collections import OrderedDict
from FML_backup.fed_kits19.dataset_creation_scripts.paths import *
from data_utils import fourier_transform_3D
# from FML_backup.fed_kits19.semi_dataset import SemiFedKiTS19
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, get_patch_size
from dataset_creation_scripts.nnunet_library.data_augmentations import transformations, semi_transformations, fixmatch_transformations, basic_transformations


class SemiKiTS19Raw(Dataset):
    """Pytorch dataset containing all the images, and segmentations for KiTS19
    Attributes
    ----------
    plan_dir : str, Where all preprocessing plans and augmentation details of the dataset are saved, used for preprocessing
    dataset_directory: where preprocessed dataset is saved
    debug: bool, Whether or not we use the dataset with only part of the features
    """
    def __init__(self, train = True, X_dtype=torch.float32, y_dtype=torch.float32,  debug=False, labelled=0, nnunet_transform = 1):
        """See description above
        Parameters
        ----------
        X_dtype : torch.dtype, optional
            Dtype for inputs `X`. Defaults to `torch.float32`.
        y_dtype : torch.dtype, optional
            Dtype for labels `y`. Defaults to `torch.int64`.
        debug : bool, optional,
            Whether or not to use only the part of the dataset downloaded in
            debug mode. Defaults to False.
        """

        plans_file = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetPlansv2.1_plans_3D.pkl'
        plans = load_pickle(plans_file)
        stage_plans = plans['plans_per_stage'][0]
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        data_aug_params = default_3D_augmentation_params
        data_aug_params['patch_size_for_spatialtransform'] = self.patch_size
        basic_generator_patch_size = get_patch_size(self.patch_size, data_aug_params['rotation_x'],
                                                    data_aug_params['rotation_y'],
                                                    data_aug_params['rotation_z'],
                                                    data_aug_params['scale_range'])

        self.pad_kwargs_data = OrderedDict()
        self.pad_mode = "constant"
        self.need_to_pad = (np.array(basic_generator_patch_size) - np.array(self.patch_size)).astype(int)

        self.weak_tr_transform, self.strong_tr_transform, self.test_transform, self.conversion_transform = semi_transformations(data_aug_params['patch_size_for_spatialtransform'],
                                                                 data_aug_params)

        self.basic_tr_transform, self.basic_test_transform = basic_transformations(data_aug_params['patch_size_for_spatialtransform'],
                                                                 data_aug_params)
        cutmix_mask_prop_range = (0.25, 0.5)
        cutmix_boxmask_n_boxes = 1
        cutmix_boxmask_fixed_aspect_ratio = False
        cutmix_boxmask_by_size = False
        cutmix_boxmask_outside_bounds = False
        cutmix_boxmask_no_invert = False
        self.mask_generator = BoxMaskGenerator(prop_range= cutmix_mask_prop_range, n_boxes= cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not  cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not  cutmix_boxmask_by_size, within_bounds=not  cutmix_boxmask_outside_bounds,
                                           invert=not  cutmix_boxmask_no_invert)
        # mask = mask_generator.generate_params(1, (2, 2))
        # self.weak_tr_transform, self.strong_tr_transform = fixmatch_transformations()

        if nnunet_transform == 1:
            self.train_transform = self.weak_tr_transform
        else:
            self.train_transform = self.basic_tr_transform

        self.dataset_directory = preprocessing_output_dir + '/Task064_KiTS_labelsFixed/nnUNetData_plans_v2.1_stage0'

        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        self.debug = debug
        self.train_test = "train" if train else "test"

        # print(self.train_test)
        self.labeled = labelled  # default is labelled
        # df = pd.read_csv('metadata/thresholded_sites.csv')
        # if self.train_test == "train":
        self.ulabeled_images = OrderedDict()

        if labelled == 1:
            df = pd.read_csv('../metadata/semi_thresholded_sites.csv')
            key = self.train_test + "_l"
            df2 = df.query("train_test_split == '" + key + "' ").reset_index(drop=True)
            self.images = df2.case_ids.tolist()
        else:
            df = pd.read_csv('../metadata/thresholded_sites.csv')
            key = self.train_test
            df2 = df.query("train_test_split == '" + key + "' ").reset_index(drop=True)
            self.images = df2.case_ids.tolist()

        # print((self.images))
        # Load image paths and properties files
        c = 0 # Case
        self.images_path = OrderedDict()
        
        for i in self.images:
            if self.train_test == "train" and self.labeled == 0:
                keyy = i
            else:
                keyy = c
                # keyy = c

            # print(i)
            self.images_path[keyy] = OrderedDict()
            self.images_path[keyy]['data_file'] = join(self.dataset_directory, "%s.npz" % i)
            self.images_path[keyy]['properties_file'] = join(self.dataset_directory, "%s.pkl" % i)
            self.images_path[keyy]['properties'] = load_pickle(self.images_path[keyy]['properties_file'])
            self.images_path[keyy]['ID'] = i # case_ID
            c += 1

        self.centers = df2.site_ids
        self.next_sample = 0
        # print(len(self.images_path))

    def get_labelled_indices(self, labeled = 1, train_test = "train"):
        if train_test == "train":
            if labeled == 1:
                key = train_test + "_l"
            else:
                key = train_test + "_u"
        else:
            key = train_test
        # df = pd.read_csv('metadata/semi_thresholded_sites.csv')
        df = pd.read_csv('../metadata/semi_thresholded_sites.csv')
        df2 = df.query("train_test_split == '" + key + "' ").reset_index(drop=True)
        images = df2.case_ids.tolist()
        print((images))
        return images 

    def get_unlabelled_indices(self, labeled = 0, train_test = "train"):
        if train_test == "train":
            if labeled == 1:
                key = train_test + "_l"
            else:
                key = train_test + "_u"
        else:
            key = train_test
        # df = pd.read_csv('metadata/semi_thresholded_sites.csv')
        df = pd.read_csv('../metadata/semi_thresholded_sites.csv')
        df2 = df.query("train_test_split == '" + key + "' ").reset_index(drop=True)
        images = df2.case_ids.tolist()
        self.ulabeled_images = df2.case_ids.tolist()
        print((images))
        return images



    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        # print("----------")
        # print(len(self.images_path))
        # print(self.images_path.keys())
        # print(idx)
        # if isfile(self.images_path[idx]['data_file'][:-4] + ".npy"):
        #     case_all_data = np.load(self.images_path[idx]['data_file'][:-4] + ".npy", memmap_mode = "r")
        # else:
        # print(idx)
        # print(self.ulabeled_images)
        # all_dict_keys = list(self.ulabeled_images.keys())
        # print(all_dict_keys)
                # else:
        # if self.train_test == "test":
        #     print(self.images_path.keys())
        #     print(idx) 
        
        case_all_data = np.load(self.images_path[idx]['data_file'])['data']

        

        # get another random data point
        # and get target_item

        properties = self.images_path[idx]['properties']


        if self.next_sample == 1:
            # print("force FG = True")
            self.next_sample = 0
            item = self.oversample_foreground_class(case_all_data, True, properties, )
            if self.train_test == 'train':
                # target_properties = self.images_path[target_index]['properties']
                target_index = random.choice(self.ulabeled_images)
                target_properties = self.images_path[target_index]['properties']
                case_all_data_target = np.load(self.images_path[target_index]['data_file'])['data']
                target_item = self.oversample_foreground_class(case_all_data_target, True, target_properties, )
                aug_img, aug_tar_img = fourier_transform_3D(item['data'] , target_item['data'], L=0.01, i=0.99)
                target_item['data'] = aug_img
        else:
            # print("force FG = False")
            self.next_sample = 1
            item = self.oversample_foreground_class(case_all_data, False, properties, )
            if self.train_test == 'train':           
                target_index = random.choice(self.ulabeled_images)
                target_properties = self.images_path[target_index]['properties']
                case_all_data_target = np.load(self.images_path[target_index]['data_file'])['data']
                target_item = self.oversample_foreground_class(case_all_data_target, True, target_properties, )

            # target_item = self.oversample_foreground_class(case_all_data_target, True, target_properties, )

        # logging.info(item['data'].shape)
        
        # print(aug_img)
        
        mask = self.mask_generator.generate_params(1, (128, 128))
        if self.train_test == 'train':
            item = self.train_transform(**item)
            target_item = self.train_transform(**target_item)
            aug_img, aug_tar_img = fourier_transform_3D(item['data'] , target_item['data'], L=0.01, i=0.99)
            target_item['data'] = aug_img
            return np.squeeze(item['data'], axis=1), np.squeeze(item['seg'], axis=1), mask, np.squeeze(target_item['data'], axis=1)
        if self.train_test == 'test':
            item = self.test_transform(**item)
            return np.squeeze(item['data'], axis=1), np.squeeze(item['seg'], axis=1)

        
        # return np.squeeze(item['data'], axis=1), np.squeeze(item['seg'], axis=1), mask, np.squeeze(target_item['data'], axis=1)


    def oversample_foreground_class(self, case_all_data, force_fg, properties,):
        #taken from nnunet
        data_shape = (1, 1, *self.patch_size)
        seg_shape = (1, 1, *self.patch_size)
        data = np.zeros(data_shape, dtype=np.float32) #shapes?
        seg = np.zeros(seg_shape, dtype=np.float32)
        need_to_pad = self.need_to_pad.copy()
        for d in range(3):
            # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides
            # always
            if need_to_pad[d] + case_all_data.shape[d + 1] < self.patch_size[d]:
                need_to_pad[d] = self.patch_size[d] - case_all_data.shape[d + 1]
        # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
        # define what the upper and lower bound can be to then sample form them with np.random.randint
        shape = case_all_data.shape[1:]
        lb_x = - need_to_pad[0] // 2
        ub_x = shape[0] + need_to_pad[0] // 2 + need_to_pad[0] % 2 - self.patch_size[0]
        lb_y = - need_to_pad[1] // 2
        ub_y = shape[1] + need_to_pad[1] // 2 + need_to_pad[1] % 2 - self.patch_size[1]
        lb_z = - need_to_pad[2] // 2
        ub_z = shape[2] + need_to_pad[2] // 2 + need_to_pad[2] % 2 - self.patch_size[2]

        # if not force_fg then we can just sample the bbox randomly from lb and ub. Else we need to make sure we get
        # at least one of the foreground classes in the patch
        if not force_fg:
            bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
            bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
            bbox_z_lb = np.random.randint(lb_z, ub_z + 1)
        else:
            # these values should have been precomputed
            if 'class_locations' not in properties.keys():
                raise RuntimeError("Please rerun the preprocessing with the newest version of nnU-Net!")

            # Foreground Classes = [0, 1]
            # this saves us a np.unique. Preprocessing already did that for all cases. Neat.
            foreground_classes = np.array(
                [i for i in properties['class_locations'].keys() if len(properties['class_locations'][i]) != 0])
            foreground_classes = foreground_classes[foreground_classes > 0]

            if len(foreground_classes) == 0:
                # this only happens if some image does not contain foreground voxels at all
                selected_class = None
                voxels_of_that_class = None
                print('case does not contain any foreground classes', i)
            else:
                selected_class = np.random.choice(foreground_classes)

                voxels_of_that_class = properties['class_locations'][selected_class]

            if voxels_of_that_class is not None:
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
                # Make sure it is within the bounds of lb and ub
                bbox_x_lb = max(lb_x, selected_voxel[0] - self.patch_size[0] // 2)
                bbox_y_lb = max(lb_y, selected_voxel[1] - self.patch_size[1] // 2)
                bbox_z_lb = max(lb_z, selected_voxel[2] - self.patch_size[2] // 2)
            else:
                # If the image does not contain any foreground classes, we fall back to random cropping
                bbox_x_lb = np.random.randint(lb_x, ub_x + 1)
                bbox_y_lb = np.random.randint(lb_y, ub_y + 1)
                bbox_z_lb = np.random.randint(lb_z, ub_z + 1)

        bbox_x_ub = bbox_x_lb + self.patch_size[0]
        bbox_y_ub = bbox_y_lb + self.patch_size[1]
        bbox_z_ub = bbox_z_lb + self.patch_size[2]

        # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
        # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
        # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
        # later
        valid_bbox_x_lb = max(0, bbox_x_lb)
        valid_bbox_x_ub = min(shape[0], bbox_x_ub)
        valid_bbox_y_lb = max(0, bbox_y_lb)
        valid_bbox_y_ub = min(shape[1], bbox_y_ub)
        valid_bbox_z_lb = max(0, bbox_z_lb)
        valid_bbox_z_ub = min(shape[2], bbox_z_ub)

        # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
        # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
        # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
        # remove label -1 in the data augmentation but this way it is less error prone)
        case_all_data = np.copy(case_all_data[:, valid_bbox_x_lb:valid_bbox_x_ub,
                                valid_bbox_y_lb:valid_bbox_y_ub,
                                valid_bbox_z_lb:valid_bbox_z_ub])

        data[0] = np.pad(case_all_data[:-1], ((0, 0),
                                              (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                              (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                              (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                         self.pad_mode, **self.pad_kwargs_data)

        seg[0] = np.pad(case_all_data[-1:], ((0, 0),
                                                (-min(0, bbox_x_lb), max(bbox_x_ub - shape[0], 0)),
                                                (-min(0, bbox_y_lb), max(bbox_y_ub - shape[1], 0)),
                                                (-min(0, bbox_z_lb), max(bbox_z_ub - shape[2], 0))),
                           'constant', **{'constant_values': -1})

        return {'data': data, 'seg': seg}

class SemiFedKiTS19(SemiKiTS19Raw):
    """
    Pytorch dataset containing for each center the features and associated labels
    for Camelyon16 federated classification.
    One can instantiate this dataset with train or test data coming from either
    of the 2 centers it was created from or all data pooled.
    The train/test split corresponds to the one from the Challenge.
    """

    def __init__(
        self,
        center=0,
        train=True,
        pooled=False,
        X_dtype=torch.float32,
        y_dtype=torch.float32,
        debug=False,
        labelled=1,
    ):
        """Instantiate the dataset
        Parameters
        pooled : bool, optional
            Whether to take all data from the 2 centers into one dataset, by
            default False
        X_dtype : torch.dtype, optional
            Dtype for inputs `X`. Defaults to `torch.float32`.
        y_dtype : torch.dtype, optional
            Dtype for labels `y`. Defaults to `torch.float32`.
        debug : bool, optional,
            Whether or not to use only the part of the dataset downloaded in
            debug mode. Defaults to False.
        augmentations: Augmentations to be applied on X
        center: Silo ID, must be from the set [0, 1, 2, 3, 4, 5]
        """
        super().__init__(X_dtype=X_dtype, train = train, y_dtype=y_dtype, debug=debug, labelled=labelled)
        # self.labeled = labelled
        if self.train_test == 'train':
            key = self.train_test + "_" + str(center)
        else:
            key = self.train_test + "_" + str(1000)
        # print(key)
        if not pooled:
            if labelled == 1:
                assert center in range(2)
            else:
                assert center in range(2, 6)
            # df = pd.read_csv('metadata/semi_thresholded_sites.csv')
            df = pd.read_csv('../metadata/thresholded_sites.csv')
            df2 = df.query("train_test_split_silo == '" + key + "' ").reset_index(drop=True)
            self.images = df2.case_ids.tolist()
            c = 0
            self.images_path = OrderedDict()
            for i in self.images:
                self.images_path[c] = OrderedDict()
                self.images_path[c]['data_file'] = join(self.dataset_directory, "%s.npz" % i)
                self.images_path[c]['properties_file'] = join(self.dataset_directory, "%s.pkl" % i)
                self.images_path[c]['properties'] = load_pickle(self.images_path[c]['properties_file'])
                self.images_path[c]['ID'] = i  # case_ID
                c += 1

            self.centers = df2.site_ids
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == "__main__":
    train_dataset = SemiKiTS19Raw( train = "train")
    labeled_idxs = train_dataset.get_labelled_indices()
    unlabeled_idxs = train_dataset.get_unlabelled_indices()
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, 2, 2-1)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler= batch_sampler, pin_memory=True)

    test_dataset = SemiFedKiTS19(1, train=False, pooled=True, labelled=1)
    test_data_global = torch.utils.data.DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=1)

    for sample in train_dataloader:
        print(sample[0].shape)
        print(sample[1].shape)
        print(sample[2])
    # train_dataset = SemiFedKiTS19(5, train=True, pooled=False, labelled=0)
    # print(' Silo 5'+str(len(train_dataset)))


    # train_dataset = SemiFedKiTS19(4, train=True, pooled=False, labelled=0)
    # print(' Silo 4 '+str(len(train_dataset)))

    # train_dataset = SemiFedKiTS19(3, train=True, pooled=False, labelled=0)
    # print(' Silo 3 '+str(len(train_dataset)))

    # train_dataset = SemiFedKiTS19(2, train=True, pooled=False, labelled=0)
    # print(' Silo 2 '+str(len(train_dataset)))

    # train_dataset = SemiFedKiTS19(1, train=True, pooled=False, labelled=1)
    # print(' Silo 1 '+str(len(train_dataset)))

    # train_dataset = SemiFedKiTS19(0, train=True, pooled=False, labelled=1)
    # print(' Silo 0 '+str(len(train_dataset)))

    # train_dataset = SemiFedKiTS19(1, train=True, pooled=True, labelled=1)
    # print(' Pool Train labelled '+str(len(train_dataset)))

    # train_dataset = SemiFedKiTS19(1, train=True, pooled=True, labelled=0)
    # print(' Pool Train unlabelled '+str(len(train_dataset)))

    # train_dataset = SemiFedKiTS19(1, train=False, pooled=True, labelled=1)
    # print(' Pool Test '+str(len(train_dataset)))


    # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

    # for sample in train_dataloader:
    #     print(sample[0].shape)
    #     print(sample[1].shape)
    #     print(sample[2])
