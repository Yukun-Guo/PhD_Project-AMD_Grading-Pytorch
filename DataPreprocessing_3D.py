"""
Data preprocessing and dataset classes for image segmentation.

This module provides dataset classes and transformation utilities for loading
and preprocessing image segmentation data, including data augmentation and
format conversion.
"""

import os
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Union

from Utils.utils import shuffle_lists, listFiles, read_image,read_csv_file, split_list
from Utils.DataAugmentation import GrayJitter, RandomCrop2D, RandomFlip


class myDataset_img(Dataset):
    """
    Custom Dataset class for image segmentation tasks.
    
    This dataset handles loading and preprocessing of image-mask pairs for
    semantic segmentation. It supports data augmentation and automatic
    format conversion.
    
    Args:
        img_list (list): List of file paths to input images
        gt_list (list): List of file paths to ground truth masks
        out_size (tuple): Target output size (height, width) for images
        shuffle (bool, optional): Whether to shuffle image-mask pairs. Defaults to True.
    
    Attributes:
        imgs (list): List of loaded image arrays
        gts (list): List of loaded ground truth arrays
        transform (transforms.Compose): Composition of data transforms
    
    Note:
        - Images are loaded as grayscale (single channel)
        - Masks are loaded as indexed images with integer class labels
        - Data augmentation includes grayscale jittering, random flip, and random crop
        - All samples are normalized to [0, 1] range and converted to tensors
    
    Example:
        >>> img_files = ['img1.png', 'img2.png']
        >>> mask_files = ['mask1.png', 'mask2.png']
        >>> dataset = myDataset_img(img_files, mask_files, (256, 256))
        >>> image, mask = dataset[0]
        >>> print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")
    """

    def __init__(self, mnv_list: List[str],fluid_list: List[str],ga_list: List[str],drusen_list: List[str], label_list: List[int], out_size: Tuple[int, int],data_type= "oct"):
        """
        Initialize the dataset with image and mask file lists.
        
        Args:
            img_list (list): List of paths to input images
            gt_list (list): List of paths to ground truth masks
            out_size (tuple): Target output size (height, width)
            shuffle (bool): Whether to shuffle the data pairs
        """
    
        self.mnv_img_list = mnv_list
        self.fluid_img_list = fluid_list
        self.ga_img_list = ga_list
        self.drusen_img_list = drusen_list
        self.labels = label_list
        
        if data_type == "oct":
            self.transform = transforms.Compose([
                # GrayJitter(),
                RandomFlip(axis=1),
                RandomCrop2D(out_size),
                Normalize(data_type="oct"),
                ToTensor()
            ])
        else:
            self.transform = transforms.Compose([
                RandomFlip(axis=1),
                RandomCrop2D(out_size),
                Normalize(data_type="bio"),
                ToTensor()
            ])

    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of image-mask pairs in the dataset
        """
        return len(self.mnv_img_list)
    
    def __getitem__(self, item: Union[int, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Get a single sample from the dataset.
        
        Args:
            item (int or torch.Tensor): Index of the sample to retrieve
        
        Returns:
            tuple: (image_tensor, mask_tensor) where:
                - image_tensor: Preprocessed image tensor of shape (C, H, W)
                - mask_tensor: Ground truth mask tensor of shape (H, W)
        """
        if torch.is_tensor(item):
            item = item.tolist()

        mnv = read_image(self.mnv_img_list[item], mode='gray')
        fluid = read_image(self.fluid_img_list[item], mode='gray')
        ga = read_image(self.ga_img_list[item], mode='gray')
        drusen = read_image(self.drusen_img_list[item], mode='gray')
        label = self.labels[item]

        mnv = np.expand_dims(mnv, axis=0)
        fluid = np.expand_dims(fluid, axis=0)
        ga = np.expand_dims(ga, axis=0)
        drusen = np.expand_dims(drusen, axis=0)

        sample = {'mnv': mnv, 'fluid': fluid, 'ga': ga, 'drusen': drusen, 'label': label}
        sample = self.transform(sample)
        
        return sample['mnv'], sample['fluid'], sample['ga'], sample['drusen'], sample['label'], self.mnv_img_list[item]

class Normalize(object):
    """
    Normalize image pixel values from [0, 255] to [0, 1] range.
    
    This transform converts uint8 pixel values to float32 values in the [0, 1] range
    by dividing by 255. The mask values are left unchanged as they represent
    class indices.
    
    Args:
        inplace (bool, optional): Whether to perform normalization in-place.
            Defaults to False.
    
    Example:
        >>> normalize = Normalize()
        >>> sample = {'img': np.array([0, 128, 255]), 'mask': np.array([0, 1, 2])}
        >>> normalized = normalize(sample)
        >>> print(normalized['img'])  # [0.0, 0.502, 1.0]
    """

    def __init__(self, inplace: bool = False,data_type='oct'):
        """
        Initialize the Normalize transform.
        
        Args:
            inplace (bool): Whether to perform normalization in-place
        """
        self.inplace = inplace
        self.data_type = data_type

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply normalization to the sample.
        
        Args:
            sample (dict): Dictionary containing 'img' and 'mask' arrays
        
        Returns:
            dict: Dictionary with normalized image and original mask
        """
        mnv, fluid, ga, drusen, label = sample['mnv'], sample['fluid'], sample['ga'], sample['drusen'], sample['label']
        if self.data_type =='oct':
            mnv = mnv / 255.0  # Normalize to [0, 1]
            fluid = fluid / 255.0
            ga = ga / 255.0
            drusen = drusen / 255.0
        else:
            drusen = np.where(drusen > 0.0, 1.0, 0.0)  # binarize drusen image, pseudo-drusen was removed
            fluid = np.where(fluid > 5.0, 1.0, 0.0)
            ga = ga / 2.0
            mnv = mnv / 255.0  # Normalize to [0, 1]
        return {'mnv': mnv, 'fluid': fluid, 'ga': ga, 'drusen': drusen, 'label': label}

    def __repr__(self) -> str:
        """Return string representation of the transform."""
        return self.__class__.__name__

class ToTensor(object):
    """
    Convert numpy arrays to PyTorch tensors.
    
    This transform converts numpy arrays to PyTorch tensors with appropriate
    data types:
    - Images: Converted to FloatTensor (for gradient computation)
    - Masks: Converted to LongTensor (for class indices)
    
    The transform maintains the array dimensions and does not change the
    data layout (assumes images are already in C x H x W format).
    
    Example:
        >>> to_tensor = ToTensor()
        >>> sample = {
        ...     'img': np.random.rand(1, 256, 256).astype(np.float32),
        ...     'mask': np.random.randint(0, 4, (256, 256)).astype(np.int64)
        ... }
        >>> tensor_sample = to_tensor(sample)
        >>> print(type(tensor_sample['img']))  # <class 'torch.Tensor'>
    """

    def __call__(self, sample: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert sample arrays to tensors.
        
        Args:
            sample (dict): Dictionary containing 'img' and 'mask' numpy arrays
        
        Returns:
            dict: Dictionary with tensor versions of image and mask
        """
        mnv, fluid, ga, drusen, label = sample['mnv'], sample['fluid'], sample['ga'], sample['drusen'], sample['label']
        
        # Convert to tensors with appropriate dtypes
        if not torch.is_tensor(mnv):
            mnv = torch.from_numpy(mnv.astype(np.float32))
        
        if not torch.is_tensor(fluid):
            fluid = torch.from_numpy(fluid.astype(np.float32))

        if not torch.is_tensor(ga):
            ga = torch.from_numpy(ga.astype(np.float32))

        if not torch.is_tensor(drusen):
            drusen = torch.from_numpy(drusen.astype(np.float32))

        # if not torch.is_tensor(label):
        #     label = torch.from_numpy(label.astype(np.int64))

        return {'mnv': mnv, 'fluid': fluid, 'ga': ga, 'drusen': drusen, 'label': label}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import toml
    config = toml.load("configs/config_oct.toml")
    image_path = config['DataModule']["data_path"]
    label_path = config['DataModule']["label_path"]
    
    data_csv = read_csv_file(label_path)
    mnv_list = [os.path.join(image_path, row['caseID'] + '_mnv.png') for row in data_csv]
    fluid_list =  [fn.replace("_mnv", "_fluid") for fn in mnv_list]
    ga_list = [fn.replace("_mnv", "_ga") for fn in mnv_list]
    drusen_list = [fn.replace("_mnv", "_drusen") for fn in mnv_list]
    label_list = [int(row['label']) for row in data_csv]
    

    if len(mnv_list) != len(label_list):
        raise ValueError(f"Mismatch: {len(mnv_list)} images vs {len(label_list)} labels")
    
    train_list, test_list, valid_list = split_list(
        list(range(len(label_list))), split=config['DataModule']["split_ratio"]
    )
        
    dataset = myDataset_img(mnv_list, fluid_list, ga_list, drusen_list, label_list, out_size=(304,304),data_type= "oct")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(len(dataloader))

    for mnv,fluid,ga,drusen,label,fname in dataloader:
        print(mnv[0].shape)
        print(fluid[0].shape)
        print(fname[0], label[0])
        bscan = make_grid(torch.cat([mnv, mnv, mnv], dim=1)).permute(1, 2, 0)
        plt.figure()
        plt.subplot(1, 3, 1), plt.imshow(bscan.numpy())
        plt.subplot(1, 3, 2), plt.imshow(np.squeeze(fluid[0].numpy()))
        plt.subplot(1, 3, 3), plt.imshow(np.squeeze(ga[0].numpy()))
        plt.show()
