"""
PyTorch Lightning Module for image segmentation models.

This module defines the neural network architecture, training logic, optimization,
and callbacks for image segmentation tasks using PyTorch Lightning framework.
"""

import toml
import torch
import torch.nn.functional as F
import torch.nn as nn
import lightning as L
import torchvision.models as models
from torchmetrics import functional as FM
from lightning.pytorch.callbacks import early_stopping, model_checkpoint, lr_monitor
from lightning.pytorch.loggers import TensorBoardLogger
from losses import dice
from torchsummary import summary
from typing import Dict, List, Any, Tuple, Optional, Union
from efficientnet_pytorch_3d import EfficientNet3D
from Utils.grad_cam import MultiInputGradCAM, MultiInputGradCAMPlusPlus, GradCAMVisualizer, ComparisonVisualizer


class NetModule(L.LightningModule):
    """
    Lightning Module for image segmentation training and inference.
    
    This class encapsulates the complete model definition including:
    - Neural network architecture (CNNNet)
    - Training and validation logic
    - Loss function computation (CrossEntropy + Dice)
    - Optimization configuration (Adam + ReduceLROnPlateau)
    - Callbacks setup (EarlyStopping, ModelCheckpoint, LRMonitor)
    - Logging configuration (TensorBoard)
    
    Args:
        config (dict): Configuration dictionary containing model settings.
            Required keys:
            - DataModule.image_shape: Input image dimensions [H, W, C]
            - DataModule.n_class: Number of segmentation classes
            - NetModule.model_name: Model name for logging and checkpoints
            - NetModule.log_dir: Directory for saving logs
            - DataModule.k_fold: Number of folds for cross-validation
    
    Attributes:
        input_size (tuple): Input image dimensions (height, width)
        img_chn (int): Number of input channels
        n_class (int): Number of output classes
        example_input_array (torch.Tensor): Example input for model summary
        out (CNNNet): The main network architecture
        model_name (str): Model name for identification
        log_dir (str): Directory for logging
        k_fold (int): Number of folds for cross-validation
    
    Example:
        >>> config = {
        ...     'DataModule': {
        ...         'image_shape': [256, 256, 1],
        ...         'n_class': 4,
        ...         'k_fold': 5
        ...     },
        ...     'NetModule': {
        ...         'model_name': 'segmentation_model',
        ...         'log_dir': './logs/'
        ...     }
        ... }
        >>> model = NetModule(config)
        >>> trainer = L.Trainer(max_epochs=100)
        >>> trainer.fit(model, datamodule=data_module)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the NetModule with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with model settings
        """
        super(NetModule, self).__init__()

        self.save_hyperparameters()
        
        self.input_size = config['DataModule']['image_shape'][:2]
        self.img_chn = config['DataModule']['image_shape'][2]
        self.n_class = config['DataModule']['n_class']
        self.lr = config['NetModule']['lr']
        self.backbone_name = config['NetModule'].get('backbone_name', 'efficientnet_b0')
        self.example_input_array = [torch.randn((1, self.img_chn, *self.input_size)),
                                    torch.randn((1, self.img_chn, *self.input_size)),
                                    torch.randn((1, self.img_chn, *self.input_size)),
                                    torch.randn((1, self.img_chn, *self.input_size))]
        self.out = self._build_2d_model(
            backbone_name=self.backbone_name,
            in_channels=self.img_chn,
            out_channels=self.n_class,
            out_activation=None
        )

        self.model_name = config['NetModule']["model_name"]
        self.log_dir = config['NetModule']["log_dir"]
        self.k_fold = config['DataModule']["k_fold"]
        self.valid_dataset = None
        self.train_dataset = None
   
    def _build_2d_model(self, backbone_name: str = 'efficientnet_b5', in_channels: Optional[int] = None, out_channels: Optional[int] = None, out_activation=None):
        """Build a simplified 2D classification head (single head, 4-class).

        The module accepts a list/tuple of four image tensors ([B, C, H, W]) and
        returns logits of shape [B, out_channels]. Uses a shared EfficientNet-B0
        backbone (features) for all branches, per-branch pooling and concatenation,
        then a small MLP head producing class logits.
        """

        in_ch = in_channels if in_channels is not None else self.img_chn
        out_ch = out_channels if out_channels is not None else self.n_class

        class Classifier(nn.Module):
            def __init__(self, in_channels, num_classes, backbone_name='efficientnet_b0'):
                super().__init__()
                # map input channels to 3 if needed
                self.to3 = nn.Conv2d(in_channels, 3, kernel_size=1) if in_channels != 3 else nn.Identity()
                self.pool = nn.AdaptiveAvgPool2d(1)
                if backbone_name == 'efficientnet_b0':
                    backbone1 = models.efficientnet_b0(weights=None)
                    backbone2 = models.efficientnet_b0(weights=None)
                    backbone3 = models.efficientnet_b0(weights=None)
                    backbone4 = models.efficientnet_b0(weights=None)
                elif backbone_name == 'efficientnet_b1':
                    backbone1 = models.efficientnet_b1(weights=None)
                    backbone2 = models.efficientnet_b1(weights=None)
                    backbone3 = models.efficientnet_b1(weights=None)
                    backbone4 = models.efficientnet_b1(weights=None)
                elif backbone_name == 'efficientnet_b2':
                    backbone1 = models.efficientnet_b2(weights=None)
                    backbone2 = models.efficientnet_b2(weights=None)
                    backbone3 = models.efficientnet_b2(weights=None)
                    backbone4 = models.efficientnet_b2(weights=None)
                elif backbone_name == 'efficientnet_b3':
                    backbone1 = models.efficientnet_b3(weights=None)
                    backbone2 = models.efficientnet_b3(weights=None)
                    backbone3 = models.efficientnet_b3(weights=None)
                    backbone4 = models.efficientnet_b3(weights=None)
                elif backbone_name == 'efficientnet_b4':
                    backbone1 = models.efficientnet_b4(weights=None)
                    backbone2 = models.efficientnet_b4(weights=None)
                    backbone3 = models.efficientnet_b4(weights=None)
                    backbone4 = models.efficientnet_b4(weights=None)
                elif backbone_name == 'efficientnet_b5':
                    backbone1 = models.efficientnet_b5(weights=None)
                    backbone2 = models.efficientnet_b5(weights=None)
                    backbone3 = models.efficientnet_b5(weights=None)
                    backbone4 = models.efficientnet_b5(weights=None)
                elif backbone_name == 'efficientnet_b6':
                    backbone1 = models.efficientnet_b6(weights=None)
                    backbone2 = models.efficientnet_b6(weights=None)
                    backbone3 = models.efficientnet_b6(weights=None)
                    backbone4 = models.efficientnet_b6(weights=None)
                elif backbone_name == 'efficientnet_b7':
                    backbone1 = models.efficientnet_b7(weights=None)
                    backbone2 = models.efficientnet_b7(weights=None)
                    backbone3 = models.efficientnet_b7(weights=None)
                    backbone4 = models.efficientnet_b7(weights=None)
                else:
                    raise ValueError(f'Unsupported backbone: {backbone_name}')

                self.backbone1 = backbone1.features
                self.backbone2 = backbone2.features
                self.backbone3 = backbone3.features
                self.backbone4 = backbone4.features
                feat_dim = 1280

                self.classifier = nn.Sequential(
                    nn.Linear(feat_dim * 4, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )

            def forward(self, inputs):                
                if not isinstance(inputs, (list, tuple)):
                    raise ValueError('TwoDClassifier expects a list/tuple of 4 input tensors')
                feats = []
                feats.append(self.pool(self.backbone1(self.to3(inputs[0]))).view(inputs[0].size(0), -1))
                feats.append(self.pool(self.backbone2(self.to3(inputs[1]))).view(inputs[1].size(0), -1))
                feats.append(self.pool(self.backbone3(self.to3(inputs[2]))).view(inputs[2].size(0), -1))
                feats.append(self.pool(self.backbone4(self.to3(inputs[3]))).view(inputs[3].size(0), -1))
                y = torch.cat(feats, dim=1)
                logits = self.classifier(y)
                return logits

        return Classifier(in_ch, out_ch,backbone_name)
    
    def forward(self, mnv: torch.Tensor, fluid: torch.Tensor, ga: torch.Tensor, drusen: torch.Tensor) -> torch.Tensor:
        """Forward pass accepting 4 separate image tensors for the classifier.
        
        Args:
            mnv (torch.Tensor): MNV image tensor [B, C, H, W]
            fluid (torch.Tensor): Fluid image tensor [B, C, H, W]  
            ga (torch.Tensor): GA image tensor [B, C, H, W]
            drusen (torch.Tensor): Drusen image tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Classification logits [B, num_classes]
        """
        return self.out([mnv, fluid, ga, drusen])
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,str], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step for one batch.
        
        Computes the forward pass, calculates loss (CrossEntropy + Dice), and logs metrics.
        
        Args:
            batch (tuple): Batch containing (images, masks) tensors
            batch_idx (int): Index of the current batch
        
        Returns:
            dict: Dictionary containing the computed loss
        """
        mnv, fluid, ga, drusen, y, _ = batch
        y_hat = self.forward(mnv, fluid, ga, drusen)
        
        train_loss = F.cross_entropy(y_hat, y)
        y_hat_argmax = torch.argmax(y_hat, dim=1)
        # train_f1 = FM.f1_score(y_hat_argmax, y, average='macro', num_classes=self.n_class,task='multiclass')
        train_acc = FM.accuracy(y_hat_argmax, y, num_classes=self.n_class,task='multiclass')
        
        # self.log("train_f1", train_f1, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", train_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", train_acc, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': train_loss}

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,str], batch_idx: int) -> None:
        """
        Validation step for one batch.
        
        Computes forward pass, calculates validation metrics (loss and IoU),
        and logs them for monitoring.
        
        Args:
            batch (tuple): Batch containing (images, masks) tensors
            batch_idx (int): Index of the current batch
        """
        mnv, fluid, ga, drusen, y, _ = batch
        y_hat = self.forward(mnv, fluid, ga, drusen)        
        # Validation loss
        val_loss = F.cross_entropy(y_hat, y)
        y_hat_argmax = torch.argmax(y_hat, dim=1)
        # val_f1 = FM.f1_score(y_hat_argmax, y, average='macro', num_classes=self.n_class,task='multiclass')
        val_acc = FM.accuracy(y_hat_argmax, y, num_classes=self.n_class,task='multiclass')
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log_dict({'val_loss': val_loss, 'val_acc': val_acc, 'lr': cur_lr}, prog_bar=True, logger=True)

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        """
        Configure optimizers and learning rate schedulers.
        
        Sets up Adam optimizer with ReduceLROnPlateau scheduler that reduces
        learning rate when validation loss plateaus.
        
        Returns:
            tuple: (optimizers, schedulers) - Lists containing optimizer and scheduler configs
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        reduce_lr_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=3,
            min_lr=1e-8
        )
        lr_scheduler = {
            'scheduler': reduce_lr_on_plateau,
            'monitor': 'val_loss',
            'interval': 'epoch',
            'reduce_on_plateau': True
        }
        return [optimizer], [lr_scheduler]

    def configure_callbacks(self) -> List[L.Callback]:
        """
        Configure training callbacks.
        
        Sets up:
        - EarlyStopping: Stops training when validation loss stops improving
        - ModelCheckpoint: Saves best model based on validation loss
        - LearningRateMonitor: Logs learning rate changes
        
        Returns:
            list: List of configured callback instances
        """
        fd = str(self.k_fold)
        early_stop = early_stopping.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-08,
            patience=10,
            verbose=True
        )

        checkpoint = model_checkpoint.ModelCheckpoint(
            dirpath=self.log_dir + self.model_name,
            monitor="val_loss",
            save_top_k=1,
            verbose=True,
            filename=f'{self.model_name}-fold={fd}-{{epoch:03d}}-{{val_loss:.5f}}'
        )

        lr_monitors = lr_monitor.LearningRateMonitor(logging_interval='epoch')
        
        return [early_stop, checkpoint, lr_monitors]

    def configure_loggers(self) -> TensorBoardLogger:
        """
        Configure training loggers.
        
        Returns:
            TensorBoardLogger: TensorBoard logger for monitoring training progress
        """
        return TensorBoardLogger(self.log_dir, name=self.model_name,log_graph=True)

    def summary(self) -> None:
        """
        Print model architecture summary using torchsummary.
        
        Automatically detects available device (CUDA or CPU) and prints
        detailed information about model layers, parameters, and memory usage.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        summary(self.to(device), tuple(self.example_input_array[0].shape[1:]))

    def create_grad_cam_visualizer(self, use_cuda: bool = True, method: str = 'gradcam',
                                  n_samples: int = 50, noise_level: float = 0.15) -> Any:
        """
        Create a gradient-based visualizer for this model.
        
        Args:
            use_cuda: Whether to use CUDA if available
            method: One of 'gradcam', 'gradcam++', 'smoothgrad', 'vargrad'
            n_samples: Number of samples for noise-based methods (SmoothGrad, VarGrad)
            noise_level: Standard deviation for Gaussian noise in noise-based methods
            
        Returns:
            Appropriate visualizer instance for this model
        """
        if method in ['gradcam', 'gradcam++']:
            return GradCAMVisualizer(self, use_cuda, method)
        elif method == 'smoothgrad':
            from Utils.grad_cam import SmoothGradVisualizer
            return SmoothGradVisualizer(self, use_cuda, n_samples, noise_level)
        elif method == 'vargrad':
            from Utils.grad_cam import VarGradVisualizer
            return VarGradVisualizer(self, use_cuda, n_samples, noise_level)
        else:
            raise ValueError(f"Unsupported method: {method}. Choose from 'gradcam', 'gradcam++', 'smoothgrad', 'vargrad'")
    
    def create_comparison_visualizer(self, use_cuda: bool = True) -> 'ComparisonVisualizer':
        """
        Create a comparison visualizer that can compare Grad-CAM and Grad-CAM++ side by side.
        
        Args:
            use_cuda: Whether to use CUDA if available
            
        Returns:
            ComparisonVisualizer instance for this model
        """
        return ComparisonVisualizer(self, use_cuda)
    
    def analyze_prediction_with_gradcam(self, mnv: torch.Tensor, fluid: torch.Tensor, 
                                      ga: torch.Tensor, drusen: torch.Tensor,
                                      target_class: Optional[int] = None,
                                      save_dir: Optional[str] = None,
                                      sample_id: str = "sample") -> Dict[str, Any]:
        """
        Analyze a prediction using Grad-CAM to show important regions.
        
        Args:
            mnv: MNV image tensor [B, C, H, W]
            fluid: Fluid image tensor [B, C, H, W]
            ga: GA image tensor [B, C, H, W] 
            drusen: Drusen image tensor [B, C, H, W]
            target_class: Target class for analysis. If None, uses predicted class
            save_dir: Directory to save visualizations
            sample_id: Identifier for this sample
            
        Returns:
            Dictionary containing prediction results and heatmaps
        """
        from pathlib import Path
        
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            logits = self.forward(mnv, fluid, ga, drusen)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())
        
        # Create Grad-CAM visualizer
        visualizer = self.create_grad_cam_visualizer()
        
        # Generate heatmaps
        inputs = [mnv, fluid, ga, drusen]
        analysis_target = target_class if target_class is not None else predicted_class
        
        save_path = Path(save_dir) if save_dir else None
        heatmaps = visualizer.analyze_sample(inputs, analysis_target, save_path, sample_id)
        
        # Cleanup
        visualizer.cleanup()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy(),
            'target_class': analysis_target,
            'heatmaps': heatmaps,
            'sample_id': sample_id
        }
    
    def analyze_prediction_with_gradcam_plus_plus(self, mnv: torch.Tensor, fluid: torch.Tensor, 
                                                 ga: torch.Tensor, drusen: torch.Tensor,
                                                 target_class: Optional[int] = None,
                                                 save_dir: Optional[str] = None,
                                                 sample_id: str = "sample") -> Dict[str, Any]:
        """
        Analyze a prediction using Grad-CAM++ to show important regions.
        
        Args:
            mnv: MNV image tensor [B, C, H, W]
            fluid: Fluid image tensor [B, C, H, W]
            ga: GA image tensor [B, C, H, W] 
            drusen: Drusen image tensor [B, C, H, W]
            target_class: Target class for analysis. If None, uses predicted class
            save_dir: Directory to save visualizations
            sample_id: Identifier for this sample
            
        Returns:
            Dictionary containing prediction results and heatmaps
        """
        from pathlib import Path
        
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            logits = self.forward(mnv, fluid, ga, drusen)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())
        
        # Create Grad-CAM++ visualizer
        visualizer = self.create_grad_cam_visualizer(use_cuda=True, method='gradcam++')
        
        # Generate heatmaps
        inputs = [mnv, fluid, ga, drusen]
        analysis_target = target_class if target_class is not None else predicted_class
        
        save_path = Path(save_dir) if save_dir else None
        heatmaps = visualizer.analyze_sample(inputs, analysis_target, save_path, sample_id)
        
        # Cleanup
        visualizer.cleanup()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy(),
            'target_class': analysis_target,
            'heatmaps': heatmaps,
            'sample_id': sample_id,
            'method': 'gradcam++'
        }
    
    def compare_gradcam_methods(self, mnv: torch.Tensor, fluid: torch.Tensor, 
                               ga: torch.Tensor, drusen: torch.Tensor,
                               target_class: Optional[int] = None,
                               save_dir: Optional[str] = None,
                               sample_id: str = "sample") -> Dict[str, Any]:
        """
        Compare Grad-CAM and Grad-CAM++ methods side by side.
        
        Args:
            mnv: MNV image tensor [B, C, H, W]
            fluid: Fluid image tensor [B, C, H, W]
            ga: GA image tensor [B, C, H, W] 
            drusen: Drusen image tensor [B, C, H, W]
            target_class: Target class for analysis. If None, uses predicted class
            save_dir: Directory to save visualizations
            sample_id: Identifier for this sample
            
        Returns:
            Dictionary containing prediction results and both sets of heatmaps
        """
        from pathlib import Path
        
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            logits = self.forward(mnv, fluid, ga, drusen)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())
        
        # Create comparison visualizer
        visualizer = self.create_comparison_visualizer()
        
        # Generate comparison analysis
        inputs = [mnv, fluid, ga, drusen]
        analysis_target = target_class if target_class is not None else predicted_class
        
        save_path = Path(save_dir) if save_dir else None
        comparison_results = visualizer.analyze_sample_comparison(inputs, analysis_target, save_path, sample_id)
        
        # Cleanup
        visualizer.cleanup()
        
        # Combine with prediction results
        comparison_results.update({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()
        })
        
        return comparison_results
    
    def analyze_prediction_with_smoothgrad(self, mnv: torch.Tensor, fluid: torch.Tensor, 
                                          ga: torch.Tensor, drusen: torch.Tensor,
                                          target_class: Optional[int] = None,
                                          save_dir: Optional[str] = None,
                                          sample_id: str = "sample",
                                          n_samples: int = 50,
                                          noise_level: float = 0.15) -> Dict[str, Any]:
        """
        Analyze a prediction using SmoothGrad to show important regions with noise reduction.
        
        Args:
            mnv: MNV image tensor [B, C, H, W]
            fluid: Fluid image tensor [B, C, H, W]
            ga: GA image tensor [B, C, H, W] 
            drusen: Drusen image tensor [B, C, H, W]
            target_class: Target class for analysis. If None, uses predicted class
            save_dir: Directory to save visualizations
            sample_id: Identifier for this sample
            n_samples: Number of noisy samples for averaging
            noise_level: Standard deviation for Gaussian noise
            
        Returns:
            Dictionary containing prediction results and heatmaps
        """
        from pathlib import Path
        from Utils.grad_cam import SmoothGradVisualizer
        
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            logits = self.forward(mnv, fluid, ga, drusen)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())
        
        # Create SmoothGrad visualizer
        visualizer = SmoothGradVisualizer(self, use_cuda=True, n_samples=n_samples, noise_level=noise_level)
        
        # Generate heatmaps
        inputs = [mnv, fluid, ga, drusen]
        analysis_target = target_class if target_class is not None else predicted_class
        
        save_path = Path(save_dir) if save_dir else None
        heatmaps = visualizer.analyze_sample(inputs, analysis_target, save_path, sample_id)
        
        # Cleanup
        visualizer.cleanup()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy(),
            'target_class': analysis_target,
            'heatmaps': heatmaps,
            'sample_id': sample_id,
            'method': 'smoothgrad',
            'n_samples': n_samples,
            'noise_level': noise_level
        }
    
    def analyze_prediction_with_vargrad(self, mnv: torch.Tensor, fluid: torch.Tensor, 
                                       ga: torch.Tensor, drusen: torch.Tensor,
                                       target_class: Optional[int] = None,
                                       save_dir: Optional[str] = None,
                                       sample_id: str = "sample",
                                       n_samples: int = 50,
                                       noise_level: float = 0.15) -> Dict[str, Any]:
        """
        Analyze a prediction using VarGrad to show regions with high gradient variance.
        
        Args:
            mnv: MNV image tensor [B, C, H, W]
            fluid: Fluid image tensor [B, C, H, W]
            ga: GA image tensor [B, C, H, W] 
            drusen: Drusen image tensor [B, C, H, W]
            target_class: Target class for analysis. If None, uses predicted class
            save_dir: Directory to save visualizations
            sample_id: Identifier for this sample
            n_samples: Number of noisy samples for variance computation
            noise_level: Standard deviation for Gaussian noise
            
        Returns:
            Dictionary containing prediction results and heatmaps
        """
        from pathlib import Path
        from Utils.grad_cam import VarGradVisualizer
        
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            logits = self.forward(mnv, fluid, ga, drusen)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())
        
        # Create VarGrad visualizer
        visualizer = VarGradVisualizer(self, use_cuda=True, n_samples=n_samples, noise_level=noise_level)
        
        # Generate heatmaps
        inputs = [mnv, fluid, ga, drusen]
        analysis_target = target_class if target_class is not None else predicted_class
        
        save_path = Path(save_dir) if save_dir else None
        heatmaps = visualizer.analyze_sample(inputs, analysis_target, save_path, sample_id)
        
        # Cleanup
        visualizer.cleanup()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy(),
            'target_class': analysis_target,
            'heatmaps': heatmaps,
            'sample_id': sample_id,
            'method': 'vargrad',
            'n_samples': n_samples,
            'noise_level': noise_level
        }
    
    def compare_all_gradient_methods(self, mnv: torch.Tensor, fluid: torch.Tensor, 
                                    ga: torch.Tensor, drusen: torch.Tensor,
                                    target_class: Optional[int] = None,
                                    save_dir: Optional[str] = None,
                                    sample_id: str = "sample",
                                    n_samples: int = 50,
                                    noise_level: float = 0.15) -> Dict[str, Any]:
        """
        Compare all gradient-based methods: Grad-CAM, Grad-CAM++, SmoothGrad, and VarGrad.
        
        Args:
            mnv: MNV image tensor [B, C, H, W]
            fluid: Fluid image tensor [B, C, H, W]
            ga: GA image tensor [B, C, H, W] 
            drusen: Drusen image tensor [B, C, H, W]
            target_class: Target class for analysis. If None, uses predicted class
            save_dir: Directory to save visualizations
            sample_id: Identifier for this sample
            n_samples: Number of noisy samples for SmoothGrad/VarGrad
            noise_level: Standard deviation for Gaussian noise
            
        Returns:
            Dictionary containing prediction results and all method heatmaps
        """
        from pathlib import Path
        from Utils.grad_cam import AllMethodsComparison
        
        self.eval()
        
        # Make prediction
        with torch.no_grad():
            logits = self.forward(mnv, fluid, ga, drusen)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_class].item())
        
        # Create all methods comparison visualizer
        visualizer = AllMethodsComparison(self, use_cuda=True, n_samples=n_samples, noise_level=noise_level)
        
        # Generate comprehensive analysis
        inputs = [mnv, fluid, ga, drusen]
        analysis_target = target_class if target_class is not None else predicted_class
        
        save_path = Path(save_dir) if save_dir else None
        all_results = visualizer.analyze_sample_all_methods(inputs, analysis_target, save_path, sample_id)
        
        # Cleanup
        visualizer.cleanup()
        
        # Combine with prediction results
        all_results.update({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy(),
            'n_samples': n_samples,
            'noise_level': noise_level
        })
        
        return all_results


if __name__ == '__main__':

    toml_file = "./configs/config_oct.toml"
    config = toml.load(toml_file)
    model = NetModule(config=config)
    model.summary()
    model.to_onnx('test.onnx')
