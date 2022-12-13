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
from copy import deepcopy
import random
import numpy as np
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.channel_selection_transforms import DataChannelSelectionTransform, \
    SegChannelSelectionTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import GammaTransform, ContrastAugmentationTransform, BrightnessTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor

from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    MaskTransform, ConvertSegmentationToRegionsTransform
from nnunet.training.data_augmentation.pyramid_augmentations import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from monai.transforms import RandomizableTransform
try:
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
except ImportError as ie:
    NonDetMultiThreadedAugmenter = None


default_3D_augmentation_params = {
    "selected_data_channels": None,
    "selected_seg_channels": None,

    "do_elastic": True,
    "elastic_deform_alpha": (0., 900.),
    "elastic_deform_sigma": (9., 13.),
    "p_eldef": 0.2,

    "do_scaling": True,
    "scale_range": (0.85, 1.25),
    "independent_scale_factor_for_each_axis": False,
    "p_independent_scale_per_axis": 1,
    "p_scale": 0.2,

    "do_rotation": True,
    "rotation_x": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_y": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_z": (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
    "rotation_p_per_axis": 1,
    "p_rot": 0.2,

    "random_crop": False,
    "random_crop_dist_to_border": None,

    "do_gamma": True,
    "gamma_retain_stats": True,
    "gamma_range": (0.7, 1.5),
    "p_gamma": 0.3,

    "do_mirror": True,
    "mirror_axes": (0, 1, 2),

    "dummy_2D": False,
    "mask_was_used_for_normalization": None,
    "border_mode_data": "constant",

    "all_segmentation_labels": None,  # used for cascade
    "move_last_seg_chanel_to_data": False,  # used for cascade
    "cascade_do_cascade_augmentations": False,  # used for cascade
    "cascade_random_binary_transform_p": 0.4,
    "cascade_random_binary_transform_p_per_label": 1,
    "cascade_random_binary_transform_size": (1, 8),
    "cascade_remove_conn_comp_p": 0.2,
    "cascade_remove_conn_comp_max_size_percent_threshold": 0.15,
    "cascade_remove_conn_comp_fill_with_other_class_p": 0.0,

    "do_additive_brightness": False,
    "additive_brightness_p_per_sample": 0.15,
    "additive_brightness_p_per_channel": 0.5,
    "additive_brightness_mu": 0.0,
    "additive_brightness_sigma": 0.1,

    "num_threads": 12 if 'nnUNet_n_proc_DA' not in os.environ else int(os.environ['nnUNet_n_proc_DA']),
    "num_cached_per_thread": 1,
}

default_2D_augmentation_params = deepcopy(default_3D_augmentation_params)
default_2D_augmentation_params["elastic_deform_alpha"] = (0., 200.)
default_2D_augmentation_params["elastic_deform_sigma"] = (9., 13.)
default_2D_augmentation_params["rotation_x"] = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_y"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)
default_2D_augmentation_params["rotation_z"] = (-0. / 360 * 2. * np.pi, 0. / 360 * 2. * np.pi)

# sometimes you have 3d data and a 3d net but cannot augment them properly in 3d due to anisotropy (which is currently
# not supported in batchgenerators). In that case you can 'cheat' and transfer your 3d data into 2d data and
# transform them back after augmentation
default_2D_augmentation_params["dummy_2D"] = False
default_2D_augmentation_params["mirror_axes"] = (0, 1)  # this can be (0, 1, 2) if dummy_2D=True


class WeakRandShiftIntensity1(RandomizableTransform):
    def randomize(self):
        super().randomize(None)
        self._offset = self.R.uniform(low=0, high=0.1)

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        return img + self._offset

class StrongRandShiftIntensity1(RandomizableTransform):
    def randomize(self):
        super().randomize(None)
        self._offset = self.R.uniform(low=0, high=0.1)

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        y = 1 - self._offset
        return img + y

class StrongRandShiftIntensity25(RandomizableTransform):
    def randomize(self):
        super().randomize(None)
        self._offset = self.R.uniform(low=0, high=0.25)

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        y = 1 - self._offset
        return img + y

class WeakRandShiftIntensity25(RandomizableTransform):
    def randomize(self):
        super().randomize(None)
        self._offset = self.R.uniform(low=0, high=0.25)

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        return img + self._offset

def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)

def transformations(patch_size, params=default_3D_augmentation_params,
                             border_val_seg=-1, pin_memory=True,
                             seeds_train=None, seeds_val=None, regions=None):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"
    tr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
        border_cval_seg=border_val_seg,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("cascade_do_cascade_augmentations") and not None and params.get(
                "cascade_do_cascade_augmentations"):
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                p_per_sample=params.get("cascade_random_binary_transform_p"),
                key="data",
                strel_size=params.get("cascade_random_binary_transform_size")))
            tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                key="data",
                p_per_sample=params.get("cascade_remove_conn_comp_p"),
                fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                dont_do_if_covers_more_than_X_percent=params.get("cascade_remove_conn_comp_fill_with_other_class_p")))

    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    tr_transforms.append(RenameTransform('seg', 'target', True))
    tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)


    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    return tr_transforms, val_transforms

def semi_transformations(patch_size, params=default_3D_augmentation_params,
                             border_val_seg=-1, pin_memory=True,
                             seeds_train=None, seeds_val=None, regions=None):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"
    tr_transforms = []
    utr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None, do_elastic_deform=params.get("do_elastic"),
        alpha=params.get("elastic_deform_alpha"), sigma=params.get("elastic_deform_sigma"),
        do_rotation=params.get("do_rotation"), angle_x=params.get("rotation_x"), angle_y=params.get("rotation_y"),
        angle_z=params.get("rotation_z"), do_scale=params.get("do_scaling"), scale=params.get("scale_range"),
        border_mode_data=params.get("border_mode_data"), border_cval_data=0, order_data=3, border_mode_seg="constant",
        border_cval_seg=border_val_seg,
        order_seg=1, random_crop=params.get("random_crop"), p_el_per_sample=params.get("p_eldef"),
        p_scale_per_sample=params.get("p_scale"), p_rot_per_sample=params.get("p_rot"),
        independent_scale_for_each_axis=params.get("independent_scale_factor_for_each_axis")
    ))
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert2DTo3DTransform())

    if params.get("do_gamma"):
        tr_transforms.append(
            GammaTransform(params.get("gamma_range"), False, True, retain_stats=params.get("gamma_retain_stats"),
                           p_per_sample=params["p_gamma"]))

    if params.get("do_mirror"):
        tr_transforms.append(MirrorTransform(params.get("mirror_axes")))

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        tr_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))
        if params.get("cascade_do_cascade_augmentations") and not None and params.get(
                "cascade_do_cascade_augmentations"):
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                p_per_sample=params.get("cascade_random_binary_transform_p"),
                key="data",
                strel_size=params.get("cascade_random_binary_transform_size")))
            tr_transforms.append(RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                channel_idx=list(range(-len(params.get("all_segmentation_labels")), 0)),
                key="data",
                p_per_sample=params.get("cascade_remove_conn_comp_p"),
                fill_with_other_class_p=params.get("cascade_remove_conn_comp_max_size_percent_threshold"),
                dont_do_if_covers_more_than_X_percent=params.get("cascade_remove_conn_comp_fill_with_other_class_p")))

    weak_tr = tr_transforms.copy()


    # Semi Supervised Transforms

    #ST transforms
    # if random.random() < 0.8:
    #     img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
    # img = transforms.RandomGrayscale(p=0.2)(img)
    # img = blur(img, p=0.5)
    # img, mask = cutout(img, mask, p=0.5)
    #
    #
    # img, mask = normalize(img, mask)


    # utr_transforms.append(ContrastAugmentationTransform())
    # utr_transforms.append(BrightnessTransform(0.0, 0.1, p_per_sample=0.15, p_per_channel=0.5))
    if random.random() < 0.8:
        utr_transforms.append(ContrastAugmentationTransform())
        utr_transforms.append(BrightnessTransform(0.0, 0.1, p_per_sample=0.15, p_per_channel=0.5)) #img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
    # No Grayscale transformation as already in grayscale
    if random.random() < 0.5:
        utr_transforms.append(GaussianBlurTransform(blur_sigma = (0.1, 2.0))) # img = blur(img, p=0.5)


    utr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.1)))
    # normalize


    if regions is not None:
        tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    # tr_transforms.append(RenameTransform('seg', 'target', True))
    # tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))

    last_transform = NumpyToTensor(['data', 'target'], 'float')

    # we already applied these transforms in weak tr
    # utr_transforms.append(RenameTransform('seg', 'target', True))
    # utr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    tr_transforms = Compose(tr_transforms)
    utr_transforms = Compose(utr_transforms)
    weak_tr = Compose(weak_tr)

    # batchgenerator_utrain = uSingleThreadedAugmenter(dataloader_utrain, weak_tr, utr_transforms)


    # # from batchgenerators.dataloading import SingleThreadedAugmenter
    # batchgenerator_ltrain = lSingleThreadedAugmenter(dataloader_ltrain, tr_transforms)
    # batchgenerator_utrain = uSingleThreadedAugmenter(dataloader_utrain, weak_tr, utr_transforms)
    # # import IPython;IPython.embed()

    # batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
    #                                               params.get("num_cached_per_thread"), seeds=seeds_train,
    #                                               pin_memory=pin_memory)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    # val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    # val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    # batchgenerator_val = lSingleThreadedAugmenter(dataloader_val, val_transforms)
    # batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
    #                                             params.get("num_cached_per_thread"), seeds=seeds_val,
    #                                             pin_memory=pin_memory)
    return tr_transforms, utr_transforms, val_transforms, last_transform


def basic_transformations(patch_size, params=default_3D_augmentation_params,
                             border_val_seg=-1, pin_memory=True,
                             seeds_train=None, seeds_val=None, regions=None):
    assert params.get('mirror') is None, "old version of params, use new keyword do_mirror"
    tr_transforms = []
    utr_transforms = []

    if params.get("selected_data_channels") is not None:
        tr_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))

    if params.get("selected_seg_channels") is not None:
        tr_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
    if params.get("dummy_2D") is not None and params.get("dummy_2D"):
        tr_transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = patch_size[1:]
    else:
        patch_size_spatial = patch_size

    if params.get("mask_was_used_for_normalization") is not None:
        mask_was_used_for_normalization = params.get("mask_was_used_for_normalization")
        tr_transforms.append(MaskTransform(mask_was_used_for_normalization, mask_idx_in_seg=0, set_outside_to=0))

    tr_transforms.append(RemoveLabelTransform(-1, 0))
    # tr_transforms.append(RenameTransform('seg', 'target', True))
    # tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    # last_transform = NumpyToTensor(['data', 'target'], 'float')
    tr_transforms = Compose(tr_transforms)


    # batchgenerator_utrain = uSingleThreadedAugmenter(dataloader_utrain, weak_tr, utr_transforms)


    # # from batchgenerators.dataloading import SingleThreadedAugmenter
    # batchgenerator_ltrain = lSingleThreadedAugmenter(dataloader_ltrain, tr_transforms)
    # batchgenerator_utrain = uSingleThreadedAugmenter(dataloader_utrain, weak_tr, utr_transforms)
    # # import IPython;IPython.embed()

    # batchgenerator_train = MultiThreadedAugmenter(dataloader_train, tr_transforms, params.get('num_threads'),
    #                                               params.get("num_cached_per_thread"), seeds=seeds_train,
    #                                               pin_memory=pin_memory)

    val_transforms = []
    val_transforms.append(RemoveLabelTransform(-1, 0))
    if params.get("selected_data_channels") is not None:
        val_transforms.append(DataChannelSelectionTransform(params.get("selected_data_channels")))
    if params.get("selected_seg_channels") is not None:
        val_transforms.append(SegChannelSelectionTransform(params.get("selected_seg_channels")))

    if params.get("move_last_seg_chanel_to_data") is not None and params.get("move_last_seg_chanel_to_data"):
        val_transforms.append(MoveSegAsOneHotToData(1, params.get("all_segmentation_labels"), 'seg', 'data'))

    # val_transforms.append(RenameTransform('seg', 'target', True))

    if regions is not None:
        val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

    # val_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    val_transforms = Compose(val_transforms)

    # batchgenerator_val = lSingleThreadedAugmenter(dataloader_val, val_transforms)
    # batchgenerator_val = MultiThreadedAugmenter(dataloader_val, val_transforms, max(params.get('num_threads') // 2, 1),
    #                                             params.get("num_cached_per_thread"), seeds=seeds_val,
    #                                             pin_memory=pin_memory)
    return tr_transforms, val_transforms

def fixmatch_transformations():

    weak_tr = WeakRandShiftIntensity1()
    strong_tr = StrongRandShiftIntensity1()

    weak_tr.set_random_state(seed=0)
    strong_tr.set_random_state(seed=0)

    return weak_tr, strong_tr