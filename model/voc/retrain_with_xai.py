import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import albumentations as albu
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM
from tqdm import tqdm
from utils import DEVICE

from model.semantic_segmentation_target import SemanticSegmentationTarget, GroundTruthSegmentationTarget
from model.voc.config import *


class VOCDataset(Dataset):
    def __init__(self, images_dir, masks_dir, classes=CLASSES, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.imgs_fps = [os.path.join(images_dir, img_id) for img_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, img_id + '.json') for img_id in self.ids]

        self.class_values = [classes.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        img = cv2.imread(self.imgs_fps[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(self.masks_fps[i], 'r') as f:
            mask_data = json.load(f)

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for object in mask_data['objects']:
            if object['classTitle'] in CLASSES:
                class_index = CLASSES.index(object['classTitle'])
                if class_index in self.class_values:
                    mask = cv2.fillPoly(mask, np.array([object['points']['exterior']]), class_index)
                else:
                    print(f"Class {object['classTitle']} not in classes list, skipping")

        if self.augmentation:
            augmented = self.augmentation(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        if self.preprocessing:
            processed = self.preprocessing(image=img, mask=mask)
            img, mask = processed['image'], processed['mask']

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask

    def __len__(self):
        return len(self.ids)


def get_activation(model, target_layer, x):
    activations = []

    def hook(module, input, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(hook)
    # Run the model forward to trigger the hook
    _ = model(x)
    # Remove the hook
    handle.remove()

    # Return the captured activations
    return activations[0]


class AAL(DiceLoss):
    def __init__(self, model, target_layer, alignment_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.target_layer = target_layer
        self.alignment_weight = alignment_weight
        self.__name__ = "AAL"

    def forward(self, y_pred, y_true, x_tensor):
        y_true = y_true.to(y_pred.device)
        x_tensor = x_tensor.to(y_pred.device)

        # Compute the standard Dice loss
        dice_loss = super().forward(y_pred, y_true)

        # Compute the Grad-CAM heatmaps using ground truth masks
        cam = HiResCAM(model=self.model, target_layers=[self.target_layer])

        # Prepare targets using ground truth masks
        targets = [GroundTruthSegmentationTarget(y_true[i]) for i in range(y_true.shape[0])]

        # Compute the Grad-CAM heatmaps
        grayscale_cams = cam(input_tensor=x_tensor, targets=targets)

        # Get activation maps from the target layer using the function
        activation_maps = get_activation(self.model, self.target_layer, x_tensor)  # Shape: (batch_size, channels, H, W)

        # Process activation maps per sample
        alignment_loss = 0.0
        for i in range(activation_maps.shape[0]):
            # Average over channels to get a single activation map per sample
            activation_map = activation_maps[i].mean(0)  # Shape: (H, W)
            # Normalize the activation map
            activation_map = (activation_map - activation_map.min()) / (
                    activation_map.max() - activation_map.min() + 1e-8)

            # Resize and normalize the Grad-CAM heatmap
            cam_resized = cv2.resize(grayscale_cams[i], (activation_map.shape[1], activation_map.shape[0]))
            cam_tensor = torch.from_numpy(cam_resized).float().to(y_pred.device)
            cam_tensor = (cam_tensor - cam_tensor.min()) / (cam_tensor.max() - cam_tensor.min() + 1e-8)

            # Compute alignment loss between activation_map and cam_tensor
            alignment_loss += F.mse_loss(activation_map, cam_tensor)

        alignment_loss /= activation_maps.shape[0]

        # Combine the Dice loss with the alignment loss
        total_loss = dice_loss + self.alignment_weight * alignment_loss
        print(f"Dice Loss: {dice_loss}, Alignment Loss: {alignment_loss}, Total Loss: {total_loss}")

        return total_loss


# Custom training and validation epochs to include input tensor
class TrainEpochWithInput(TrainEpoch):
    def batch_update(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        prediction = self.model(x)
        loss = self.loss(prediction, y, x)
        loss.backward()
        self.optimizer.step()

        return loss, prediction


class ValidEpochWithInput(ValidEpoch):
    def batch_update(self, x, y):
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x)

        # Enable gradient tracking for GradCAM
        with torch.enable_grad():
            prediction_for_gradcam = self.model(x)
            loss = self.loss(prediction_for_gradcam, y, x)

        return loss, prediction


class EarlyStopping:
    def __init__(self, optimizer, patience=7, verbose=False, delta=0.005):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.optimizer = optimizer

    def __call__(self, val_loss, model, path_name):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path_name)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter == 5:
                self.optimizer.param_groups[0]['lr'] = 1e-5
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path_name):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path_name)
        self.val_loss_min = val_loss


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def get_training_augmentation():
    train_transform = [
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16),
        albu.Resize(height=256, width=256)
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=16, pad_width_divisor=16),
        albu.Resize(height=256, width=256)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    if x.ndim == 2:  # If the input is a 2D array (e.g., mask), add a channel dimension
        x = np.expand_dims(x, axis=2)
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


if __name__ == '__main__':
    # Initialize the model
    model = smp.DeepLabV3Plus(encoder_name=ENCODER,
                              encoder_weights=ENCODER_WEIGHTS,
                              classes=len(CLASSES),
                              activation=ACTIVATIONS)
    model.to(DEVICE)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    # Dataset and DataLoader setup
    train_dataset = VOCDataset(x_train_dir, y_train_dir, classes=CLASSES,
                               augmentation=get_training_augmentation(),
                               preprocessing=get_preprocessing(preprocessing_fn))
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=12)
    val_dataset = VOCDataset(x_val_dir, y_val_dir, classes=CLASSES,
                             augmentation=get_validation_augmentation(),
                             preprocessing=get_preprocessing(preprocessing_fn))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    print('Number of samples in train dataset:', len(train_dataset))
    print('Number of samples in val dataset:', len(val_dataset))

    # Loss, Metrics, and Optimizer
    # loss = XAIEnhancedDiceLoss(model=model, target_layer=model.decoder.block1, category_idx=0, mode='multiclass')

    # Initialize the loss function
    # loss = WeightedDiceLoss(model=model, target_layer=model.decoder.block1, category_idx=0, mode='multiclass')
    loss = AAL(
        model=model,
        target_layer=model.decoder.block1,  # Your target layer
        alignment_weight=0.2,
        mode='multiclass'
    )

    metrics = [IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])

    # Early Stopping Initialization
    early_stopping = EarlyStopping(patience=15, verbose=True)

    train_epoch = TrainEpochWithInput(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    val_epoch = ValidEpochWithInput(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # Lists to store the logs
    train_loss_log = []
    train_iou_log = []
    val_loss_log = []
    val_iou_log = []

    for epoch in tqdm(range(EPOCHS)):
        train_logs = train_epoch.run(train_loader)

        # Save the logs
        train_loss_log.append(train_logs['AAL'])
        train_iou_log.append(train_logs['iou_score'])

        # Validation
        val_logs = val_epoch.run(val_loader)

        # Save the logs
        val_loss_log.append(val_logs['AAL'])
        val_iou_log.append(val_logs['iou_score'])

        print(
            f"Epoch: {epoch}, Train Loss: {train_logs['AAL']}, Train IoU: {train_logs['iou_score']}, "
            f"Val Loss: {val_logs['AAL']}, Val IoU: {val_logs['iou_score']}")

        # Early Stopping
        early_stopping(val_logs['AAL'], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Save the logs for future use
    np.save('train_loss_log_aug_xai_reg.npy', train_loss_log)
    np.save('train_iou_log_aug_xai_reg.npy', train_iou_log)
    np.save('val_loss_log_aug_xai_reg.npy', val_loss_log)
    np.save('val_iou_log_aug_xai_reg.npy', val_iou_log)
    print(f'Train IOU: {train_iou_log}, Val IOU: {val_iou_log}')
