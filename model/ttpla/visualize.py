import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, XGradCAM, AblationCAM, EigenGradCAM, LayerCAM
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from torch.utils.data import DataLoader
import json
import time
import matplotlib.pyplot as plt
from model.ttpla.config import *
from model.ttpla.retrain_with_xai import TTPLADataset, get_preprocessing, \
    get_validation_augmentation
from utils import DEVICE

num_classes = len(CLASSES)


# %%
def compute_gradcam_heatmap(model, target_layer, input_tensor, masks):
    # input_tensor.requires_grad = True  # Already set in the forward method
    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = []
    for i in range(input_tensor.size(0)):
        target = SemanticSegmentationTarget(category=None, mask=masks[i])
        targets.append(target)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam


# %%
def get_activation(model, target_layer, x):
    activations = []

    def hook(module, input, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(hook)
    _ = model(x)
    handle.remove()
    return activations[0]


# %%
def normalize_heatmap(heatmap):
    # Compute the minimum and maximum values along the specified dimensions
    heatmap_min = heatmap.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    heatmap_max = heatmap.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]

    # Normalize the heatmap using broadcasting
    normalized_heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min + 1e-8)
    return normalized_heatmap


# %%
class EarlyStopping:
    def __init__(self, optimizer, patience=7, verbose=False, delta=0):
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


# %%
class GradCAMDistillationLossAlpha(torch.nn.Module):
    def __init__(self, student_loss_fn, teacher_model, student_model, distillation_weight, soft_label_weight,
                 temperature=1.0):
        super(GradCAMDistillationLossAlpha, self).__init__()
        self.student_loss_fn = student_loss_fn
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distillation_weight = distillation_weight
        self.soft_label_weight = soft_label_weight
        self.temperature = temperature
        self.kl_div_loss_fn = torch.nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_output, y_true, x_tensor, compute_gradcam=True):
        # Compute the student's segmentation loss
        student_loss = self.student_loss_fn(student_output, y_true)

        # Obtain teacher's outputs (soft labels)
        with torch.no_grad():
            teacher_output = self.teacher_model(x_tensor)

        # Compute the soft label loss (KL Divergence)
        # Apply temperature scaling
        student_logits = student_output / self.temperature
        teacher_logits = teacher_output / self.temperature

        # Compute probabilities
        student_log_probs = F.log_softmax(student_logits, dim=1)
        teacher_probs = F.softmax(teacher_logits, dim=1)

        # Compute KL Divergence
        soft_label_loss = self.kl_div_loss_fn(student_log_probs, teacher_probs) * (self.temperature ** 2)

        distillation_loss = 0.0  # Default value

        if compute_gradcam:
            # Ensure x_tensor requires gradients
            x_tensor.requires_grad = True

            # Compute Grad-CAM heatmaps
            # Student masks (student predictions)
            student_masks = torch.argmax(student_output, dim=1).float()

            # Teacher masks (teacher predictions)
            teacher_masks = torch.argmax(teacher_output, dim=1).float()

            # Compute Grad-CAM heatmaps for teacher and student
            teacher_heatmap = compute_gradcam_heatmap(
                model=self.teacher_model,
                target_layer=self.teacher_model.decoder.block1,
                input_tensor=x_tensor,
                masks=teacher_masks
            )

            student_heatmap = compute_gradcam_heatmap(
                model=self.student_model,
                target_layer=self.student_model.decoder.conv,
                input_tensor=x_tensor,
                masks=student_masks
            )


            # Convert heatmaps to torch tensors and move to device
            teacher_heatmap = torch.from_numpy(teacher_heatmap).float().to(student_output.device)
            student_heatmap = torch.from_numpy(student_heatmap).float().to(student_output.device)

            # Normalize heatmaps
            teacher_heatmap = normalize_heatmap(teacher_heatmap)
            student_heatmap = normalize_heatmap(student_heatmap)

            # Compute distillation loss between heatmaps
            distillation_loss = F.mse_loss(student_heatmap, teacher_heatmap)

        # Combine the losses
        total_loss = (
                student_loss
                + self.soft_label_weight * soft_label_loss
                + self.distillation_weight * distillation_loss
        )

        return total_loss


# %%
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        # Ensure mask is a torch tensor
        if isinstance(mask, np.ndarray):
            self.mask = torch.from_numpy(mask).float()
        else:
            self.mask = mask.float()
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        # model_output shape: (C, H, W)
        if self.category is not None:
            output = model_output[self.category, :, :]
        else:
            output = model_output.sum(dim=0)
        loss = (output * self.mask).sum()
        # Add epsilon to prevent zero gradients
        loss += 1e-6
        return loss


# %%
class NamedDiceLoss(DiceLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = "DiceLoss"


# %%
def evaluate_model(model, dataloader):
    model.eval()
    test_loss = 0
    test_iou = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Get predictions
            output = model(x_batch)

            # Compute loss (use DiceLoss for evaluation)
            loss = DiceLoss(mode='multiclass')(output, y_batch)

            # Accumulate metrics
            test_loss += loss.item()
            y_pred = torch.argmax(output, dim=1)
            y_true = torch.argmax(y_batch, dim=1)
            iou = IoU()(y_pred, y_true)
            test_iou += iou.item()

    # Compute average loss and IoU
    test_loss /= len(dataloader)
    test_iou /= len(dataloader)

    return test_loss, test_iou


# %%
# Define preprocessing function
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Datasets and dataloaders
batch_size = 1  # Set batch size to 1 for IoU calculations as in the original code

# Train dataset and dataloader
train_dataset = TTPLADataset(
    x_train_dir,
    y_train_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Validation dataset and dataloader
val_dataset = TTPLADataset(
    x_val_dir,
    y_val_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Test dataset and dataloader
test_dataset = TTPLADataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def calculate_iou(model, dataloader, device, classes):
    iou_metric = IoU(threshold=0.5)
    iou_scores = {cls: [] for cls in classes}

    with torch.no_grad():
        for i, (image, gt_mask) in enumerate(dataloader):
            image, gt_mask = image.to(device), gt_mask.to(device)
            pr_mask = model(image).squeeze().cpu().numpy()
            pr_mask = np.argmax(pr_mask, axis=0)

            gt_mask = gt_mask.squeeze().cpu().numpy()

            if pr_mask.shape != gt_mask.shape:
                print(f"Shape mismatch: pr_mask shape {pr_mask.shape}, gt_mask shape {gt_mask.shape}")
                continue

            for idx, cls in enumerate(classes):
                gt_mask_filtered = (gt_mask == idx).astype(float)
                pr_mask_filtered = (pr_mask == idx).astype(float)

                gt_mask_tensor = torch.tensor(gt_mask_filtered, device=device, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0)
                pr_mask_tensor = torch.tensor(pr_mask_filtered, device=device, dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0)

                iou_score = iou_metric(pr_mask_tensor, gt_mask_tensor).item()
                iou_scores[cls].append(iou_score)

    avg_iou_scores = {cls: np.mean(scores) for cls, scores in iou_scores.items()}
    miou = np.mean(list(avg_iou_scores.values()))

    return avg_iou_scores, miou


def evaluate_model_metric(model, dataloader, device, classes, dataset_name):
    avg_iou_scores, miou = calculate_iou(model, dataloader, device, classes)

    print(f"\nIoU Scores for each category in {dataset_name}:")
    for cls, score in avg_iou_scores.items():
        print(f"{cls}: {score:.4f}")
    print(f"Mean IoU for {dataset_name}: {miou:.4f}\n")


# Function to load the model from a given checkpoint path
def load_model(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(DEVICE)
    model.eval()
    return model

# Denormalize function (if input is normalized) for better visualization
def denormalize(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    image_tensor = image_tensor.clone().detach().cpu()
    for t, m, s in zip(image_tensor, mean, std):
        t.mul_(s).add_(m)  # Denormalize each channel
    return image_tensor.clamp(0, 1)  # Clip to valid range

def compute_mean_iou_per_sample(pred_masks, gt_masks, num_classes):
    """
    Computes the mean IoU per sample over all classes.

    Args:
        pred_masks (torch.Tensor): Predicted masks of shape [batch_size, H, W].
        gt_masks (torch.Tensor): Ground truth masks of shape [batch_size, H, W].
        num_classes (int): Number of classes.

    Returns:
        List[float]: Mean IoU for each sample in the batch.
    """
    batch_size = pred_masks.size(0)
    iou_scores = []

    for i in range(batch_size):
        pred = pred_masks[i]  # [H, W]
        gt = gt_masks[i]  # [H, W]

        iou_per_class = []
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            gt_cls = (gt == cls)

            intersection = (pred_cls & gt_cls).sum().item()
            union = (pred_cls | gt_cls).sum().item()
            if union == 0:
                iou = float('nan')  # Avoid division by zero
            else:
                iou = intersection / union
            iou_per_class.append(iou)

        # Mean IoU over classes for this sample
        mean_iou = np.nanmean(iou_per_class)
        iou_scores.append(mean_iou)

    return iou_scores

# Path to your meta.json
meta_json_path = '/home/r6639/Projects/xaiseg/data/ttpla/meta.json'  # Replace with the actual path

# Load meta.json and create color_map
with open(meta_json_path, 'r') as f:
    meta = json.load(f)

classes = meta['classes']

# Create color_map starting with background
color_map = [(0, 0, 0)]  # Background color
for cls in classes:
    hex_color = cls['color']
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
    color_map.append(rgb_color)

# Function to apply the color map
def apply_custom_colormap(mask, color_map):
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx, color in enumerate(color_map):
        color_mask[mask == class_idx] = color
    return color_mask

# %%
# Initialize the student model
ENCODER_TEACHER = 'resnet101'
teacher_model_base = smp.DeepLabV3Plus(
    encoder_name=ENCODER_TEACHER,
    encoder_weights=ENCODER_WEIGHTS,  # You can set this to None if you want to train from scratch
    classes=len(CLASSES),
    activation=ACTIVATIONS
)

# Initialize the student model
ENCODER_STUDENT = 'resnet18'
# student_model_base = smp.DeepLabV3Plus(
#     encoder_name=ENCODER_STUDENT,
#     encoder_weights=ENCODER_WEIGHTS,  # You can set this to None if you want to train from scratch
#     classes=len(CLASSES),
#     activation=ACTIVATIONS
# )
# student_model_base_no_cam = smp.DeepLabV3Plus(
#     encoder_name=ENCODER_STUDENT,
#     encoder_weights=ENCODER_WEIGHTS,  # You can set this to None if you want to train from scratch
#     classes=len(CLASSES),
#     activation=ACTIVATIONS
# )
student_model_base_cam = smp.DeepLabV3Plus(
    encoder_name=ENCODER_STUDENT,
    encoder_weights=ENCODER_WEIGHTS,  # You can set this to None if you want to train from scratch
    classes=len(CLASSES),
    activation=ACTIVATIONS
)

# Load the student and teacher models
# student_model_no_no = load_model(student_model_base, 'student_model_1_1_01.pth')
student_model_cam = load_model(student_model_base_cam, 'student_model_1_1_01.pth')
# student_model_no_cam = load_model(student_model_base_no_cam, 'student_model_GradCAMPP_1_1_01.pth')
teacher_model = load_model(teacher_model_base, 'teacher_model.pth')

datasets = {
    "Training Set": train_loader,
    "Validation Set": val_loader,
    "Test Set": test_loader
}

# Visualization loop
i = 0
for x_sample, y_sample in test_loader:
    if i == 50:
        break

    x_sample = x_sample.to(DEVICE)
    y_sample = y_sample.to(DEVICE)
    x_sample.requires_grad = True  # Enable gradients for Grad-CAM if necessary

    # Generate predictions and masks for all models
    with torch.no_grad():
        # student_output_no_no = student_model_no_no(x_sample)
        student_output_cam = student_model_cam(x_sample)
        # student_output_no_cam = student_model_no_cam(x_sample)
        teacher_output = teacher_model(x_sample)

    # Generate masks by taking the argmax along the class dimension
    # student_masks_no_no = torch.argmax(student_output_no_no, dim=1, keepdim=True).float().squeeze(1)
    student_masks_cam = torch.argmax(student_output_cam, dim=1, keepdim=True).float().squeeze(1)
    # student_masks_no_cam = torch.argmax(student_output_no_cam, dim=1, keepdim=True).float().squeeze(1)
    teacher_masks = torch.argmax(teacher_output, dim=1, keepdim=True).float().squeeze(1)

    # Ground truth mask
    gt_masks = y_sample.squeeze(1)  # [batch_size, H, W]

    # Compute per-sample mean IoU for each model
    # iou_no_no = compute_mean_iou_per_sample(student_masks_no_no, gt_masks, num_classes)
    iou_cam = compute_mean_iou_per_sample(student_masks_cam, gt_masks, num_classes)
    # iou_no_cam = compute_mean_iou_per_sample(student_masks_no_cam, gt_masks, num_classes)
    iou_teacher = compute_mean_iou_per_sample(teacher_masks, gt_masks, num_classes)

    # Compute Grad-CAM heatmaps for all models
    # student_heatmap_no_no = compute_gradcam_heatmap(
    #     model=student_model_no_no,
    #     target_layer=student_model_no_no.decoder.block1,
    #     input_tensor=x_sample,
    #     masks=student_masks_no_no
    # )
    start_time = time.time()
    student_heatmap_cam = compute_gradcam_heatmap(
        model=student_model_cam,
        target_layer=student_model_cam.decoder.block1,
        input_tensor=x_sample,
        masks=student_masks_cam
    )
    end_time = time.time()
    print(f"Time taken to compute heatmap: {end_time - start_time:.4f} seconds")

    # student_heatmap_no_cam = compute_gradcam_heatmap(
    #     model=student_model_no_cam,
    #     target_layer=student_model_no_cam.decoder.block1,
    #     input_tensor=x_sample,
    #     masks=student_masks_no_cam
    # )

    teacher_heatmap = compute_gradcam_heatmap(
        model=teacher_model,
        target_layer=teacher_model.decoder.block1,
        input_tensor=x_sample,
        masks=teacher_masks
    )

    # Convert heatmaps to tensors
    # student_heatmap_no_no = torch.from_numpy(student_heatmap_no_no).float().to(DEVICE)
    student_heatmap_cam = torch.from_numpy(student_heatmap_cam).float().to(DEVICE)
    # student_heatmap_no_cam = torch.from_numpy(student_heatmap_no_cam).float().to(DEVICE)
    teacher_heatmap = torch.from_numpy(teacher_heatmap).float().to(DEVICE)

    plt.figure(figsize=(20, 15))

    # Original image
    plt.subplot(4, 2, 1)
    original_image = denormalize(x_sample.squeeze())
    plt.imshow(np.transpose(original_image.cpu().numpy(), (1, 2, 0)))  # Convert CHW to HWC
    plt.title("Original Image")
    plt.axis("off")

    # Ground truth mask
    plt.subplot(4, 2, 2)
    gt_colored_mask = apply_custom_colormap(gt_masks[0].cpu().numpy().astype(int), color_map)
    plt.imshow(gt_colored_mask)
    plt.title("Ground Truth Mask")
    plt.axis("off")

    # Student mask with CAM
    plt.subplot(4, 2, 3)
    student_colored_cam = apply_custom_colormap(student_masks_cam.squeeze(0).cpu().numpy().astype(int), color_map)
    plt.imshow(student_colored_cam)
    plt.title(f"Student \nIoU: {iou_cam[0]:.4f}")
    plt.axis("off")

    # Grad-CAM heatmap for student with CAM
    plt.subplot(4, 2, 4)
    plt.imshow(student_heatmap_cam.squeeze(0).cpu().numpy(), cmap='jet')  # Keep jet for Grad-CAM
    plt.title("CAM Heatmap (Student)")
    plt.axis("off")

    # Teacher mask
    plt.subplot(4, 2, 5)
    teacher_colored_mask = apply_custom_colormap(teacher_masks.squeeze(0).cpu().numpy().astype(int), color_map)
    plt.imshow(teacher_colored_mask)
    plt.title(f"Teacher \nIoU: {iou_teacher[0]:.4f}")
    plt.axis("off")

    # Grad-CAM heatmap for teacher
    plt.subplot(4, 2, 6)
    plt.imshow(teacher_heatmap.squeeze(0).cpu().numpy(), cmap='jet')
    plt.title("CAM Heatmap (Teacher)")
    plt.axis("off")


    plt.tight_layout()
    # Save plot to results folder
    plt.savefig(f"results/GradCAM/visualization_{i}.png")
    i += 1