"""
PyTorch Lightning DataModule for image segmentation tasks.

This module provides a comprehensive data handling solution for image segmentation,
including data loading, preprocessing, splitting, and DataLoader creation.
"""
import os
import lightning as L
from typing import Optional
from torch.utils.data import DataLoader
from DataPreprocessing import myDataset_img
from Utils.utils import listFiles, split_list, read_csv_file


class DataModel(L.LightningDataModule):
    """
    Lightning DataModule for image segmentation data handling.
    
    This class manages all data-related operations including:
    - Loading image and mask file lists
    - Splitting data into train/validation/test sets
    - Creating dataset instances with appropriate transforms
    - Providing DataLoaders for training, validation, and testing
    
    Args:
        config (dict): Configuration dictionary containing DataModule settings.
            Required keys:
            - image_path (str): Path to directory containing input images
            - mask_path (str): Path to directory containing ground truth masks
            - batch_size (int): Batch size for DataLoaders
            - image_shape (list): Target image dimensions [height, width, channels]
            - shuffle (bool): Whether to shuffle training data
            - split_ratio (list): Data split ratios [train, test, validation]
    
    Attributes:
        image_path (str): Path to input images directory
        mask_path (str): Path to ground truth masks directory
        batch_size (int): Batch size for data loaders
        img_shape (list): Image dimensions [height, width, channels]
        shuffle (bool): Whether to shuffle training data
        split_ratio (list): Data split ratios
        img_size (tuple): Image size as (height, width)
        train_dataset: Training dataset instance
        valid_dataset: Validation dataset instance
        test_dataset: Test dataset instance
    
    Example:
        >>> config = {
        ...     'DataModule': {
        ...         'image_path': './data/images',
        ...         'mask_path': './data/masks',
        ...         'batch_size': 16,
        ...         'image_shape': [256, 256, 1],
        ...         'shuffle': True,
        ...         'split_ratio': [0.7, 0.2, 0.1]
        ...     }
        ... }
        >>> data_module = DataModel(config)
        >>> data_module.setup()
        >>> train_loader = data_module.train_dataloader()
    """

    def __init__(self, config: dict):
        """
        Initialize the DataModel with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with DataModule settings
        """
        super().__init__()
        self.image_path = config['DataModule']["data_path"]
        self.label_path = config['DataModule']["label_path"]
        self.batch_size = config['DataModule']["batch_size"]
        self.img_shape = config['DataModule']["image_shape"]
        self.shuffle = config['DataModule']["shuffle"]
        self.split_ratio = config['DataModule']["split_ratio"]
        self.img_size = self.img_shape[:2]  # (H, W)
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Prepare data for training (called only on 1 GPU/TPU in distributed training).
        
        This method is called only once and is used for downloading or preparing
        data that should not be done in parallel. Currently not implemented as
        data preparation is handled in setup().
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training, validation, and testing.
        
        This method is called on every GPU in distributed training and is responsible
        for loading data files, splitting them according to the configuration, and
        creating dataset instances.
        
        Args:
            stage (str, optional): Current stage of training. Can be 'fit', 'validate',
                'test', or None. Defaults to None.
        
        Side Effects:
            - Sets self.train_dataset, self.valid_dataset, and self.test_dataset
            - Prints dataset sizes to console
        
        Raises:
            FileNotFoundError: If image or mask directories don't exist
            ValueError: If number of images and masks don't match
        """

        data_csv = read_csv_file(self.label_path)
        self.mnv_list = [os.path.join(self.image_path, row['caseID'] + '_mnv.png') for row in data_csv]
        self.fluid_list =  [fn.replace("_mnv", "_fluid") for fn in self.mnv_list]
        self.ga_list = [fn.replace("_mnv", "_ga") for fn in self.mnv_list]
        self.drusen_list = [fn.replace("_mnv", "_drusen") for fn in self.mnv_list]
        self.label_list = [int(row['label']) for row in data_csv]
        

        if len(self.mnv_list) != len(self.label_list):
            raise ValueError(f"Mismatch: {len(self.mnv_list)} images vs {len(self.label_list)} labels")
        
        self.train_list, self.valid_list, self.test_list = split_list(
            list(range(len(self.label_list))), split=self.split_ratio
        )
        
        train_mnv_list = [self.mnv_list[i] for i in self.train_list]
        train_fluid_list = [self.fluid_list[i] for i in self.train_list]
        train_ga_list = [self.ga_list[i] for i in self.train_list]
        train_drusen_list = [self.drusen_list[i] for i in self.train_list]
        train_label_list = [self.label_list[i] for i in self.train_list]
        valid_mnv_list = [self.mnv_list[i] for i in self.valid_list]
        valid_fluid_list = [self.fluid_list[i] for i in self.valid_list]
        valid_ga_list = [self.ga_list[i] for i in self.valid_list]
        valid_drusen_list = [self.drusen_list[i] for i in self.valid_list]
        valid_label_list = [self.label_list[i] for i in self.valid_list]
        test_mnv_list = [self.mnv_list[i] for i in self.test_list]
        test_fluid_list = [self.fluid_list[i] for i in self.test_list]
        test_ga_list = [self.ga_list[i] for i in self.test_list]
        test_drusen_list = [self.drusen_list[i] for i in self.test_list]
        test_label_list = [self.label_list[i] for i in self.test_list]
        
        self.train_dataset = myDataset_img(train_mnv_list, train_fluid_list, train_ga_list, train_drusen_list, train_label_list, self.img_size)
        self.valid_dataset = myDataset_img(valid_mnv_list, valid_fluid_list, valid_ga_list, valid_drusen_list, valid_label_list, self.img_size)
        self.test_dataset = myDataset_img(test_mnv_list, test_fluid_list, test_ga_list, test_drusen_list, test_label_list, self.img_size)
        
        print(f'Train on {len(self.train_dataset)} samples, '
              f'validation on {len(self.valid_dataset)} samples, '
              f'test on {len(self.test_dataset)} samples.')

    def train_dataloader(self) -> DataLoader:
        """
        Create and return the training DataLoader.
        
        Returns:
            DataLoader: Training data loader with shuffling enabled
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=2,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        Create and return the validation DataLoader.
        
        Returns:
            DataLoader: Validation data loader without shuffling
        """
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """
        Create and return the test DataLoader.
        
        Returns:
            DataLoader: Test data loader without shuffling
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

    def teardown(self, stage: Optional[str] = None):
        """
        Clean up after training/testing is complete.
        
        Args:
            stage (str, optional): Current stage being torn down
        """
        pass

if __name__ == "__main__":
    import toml

    toml_file = "./configs/config_oct.toml"
    config = toml.load(toml_file)

    data_model = DataModel(config=config)
    data_model.setup()

    train_loader = data_model.train_dataloader()
    for batch in train_loader:
        mnv, fluid, ga, drusen, y, fname = batch
        print(mnv.shape, fluid.shape, ga.shape, drusen.shape, y, fname)