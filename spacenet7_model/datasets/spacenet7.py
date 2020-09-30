import json

import numpy as np

from skimage import io
from torch.utils.data import Dataset


class SpaceNet7Dataset(Dataset):
    CLASSES = [
        'building_footprint',  # 1st (R) channel in mask
        'building_boundary',  # 2nd (G) channel in mask
        'building_contact',  # 3rd (B) channel in mask
    ]

    def __init__(self,
                 config,
                 data_list_path,
                 augmentation=None,
                 preprocessing=None):
        """[summary]

        Args:
            config ([type]): [description]
            data_list_path ([type]): [description]
            augmentation ([type], optional): [description]. Defaults to None.
            preprocessing ([type], optional): [description]. Defaults to None.
        """
        # generate full path to image/label files
        with open(data_list_path) as f:
            data_list = json.load(f)

        self.image_paths, self.mask_paths = [], []
        for data in data_list:
            self.image_paths.append(data['image_masked'])
            self.mask_paths.append(data['building_mask'])

        # convert str names to class values on masks
        classes = config.INPUT.CLASSES
        if not classes:
            # if classes is empty, use all classes
            classes = self.CLASSES
        self.class_values = [self.CLASSES.index(c) for c in classes]

        self.device = config.MODEL.DEVICE

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.in_channels = config.MODEL.IN_CHANNELS
        assert self.in_channels in [3, 4]

    def __getitem__(self, i):
        """[summary]

        Args:
            i ([type]): [description]

        Returns:
            [type]: [description]
        """
        image = io.imread(self.image_paths[i])
        mask = io.imread(self.mask_paths[i])

        if self.in_channels == 3:
            # remove alpha channel
            image = image[:, :, :3]
        _, _, c = image.shape
        assert c == self.in_channels

        # extract certain classes from mask
        masks = [(mask[:, :, v] > 0) for v in self.class_values]
        mask = np.stack(masks,
                        axis=-1).astype('float')  # XXX: multi class setting.

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.image_paths)


class SpaceNet7TestDataset(Dataset):
    def __init__(self,
                 config,
                 image_paths,
                 augmentation=None,
                 preprocessing=None):
        """[summary]

        Args:
            config ([type]): [description]
            image_paths ([type]): [description]
            augmentation ([type], optional): [description]. Defaults to None.
            preprocessing ([type], optional): [description]. Defaults to None.
        """
        self.image_paths = image_paths

        self.device = config.MODEL.DEVICE

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.in_channels = config.MODEL.IN_CHANNELS
        assert self.in_channels in [3, 4]

    def __getitem__(self, i):
        """[summary]

        Args:
            i ([type]): [description]

        Returns:
            [type]: [description]
        """
        image_path = self.image_paths[i]
        image = io.imread(image_path)

        if self.in_channels == 3:
            # remove alpha channel
            image = image[:, :, :3]
        _, _, c = image.shape
        assert c == self.in_channels

        original_shape = image.shape

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return {
            'image': image,
            'image_path': image_path,
            'original_shape': original_shape,
        }

    def __len__(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.image_paths)
