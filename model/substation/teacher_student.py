import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.utils.metrics import IoU
from segmentation_models_pytorch.utils.train import TrainEpoch, ValidEpoch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.semantic_segmentation_target import SemanticSegmentationTarget
from model.substation.config import *
from model.substation.retrain_with_xai import SubstationDataset, get_training_augmentation, get_preprocessing, \
    get_validation_augmentation
from utils import DEVICE


def compute_gradcam_heatmap(model, target_layer, input_tensor, target_category):
    """
    Computes the GradCAM heatmap for a given model and input.
    """
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor, targets=[SemanticSegmentationTarget(target_category)])
    # grayscale_cam has shape (N, H, W)
    return grayscale_cam


class DistillationLoss(torch.nn.Module):
    def __init__(self, student_loss_fn, model, teacher_model, target_layer, distillation_weight=0.1):
        super(DistillationLoss, self).__init__()
        self.student_loss_fn = student_loss_fn
        self.model = model
        self.teacher_model = teacher_model
        self.target_layer = target_layer
        self.distillation_weight = distillation_weight

    def forward(self, student_output, teacher_output, y_true, x_tensor):
        # Compute the student's segmentation loss
        student_loss = self.student_loss_fn(student_output, y_true)

        # Compute GradCAM heatmaps for both teacher and student models
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_cam = compute_gradcam_heatmap(self.teacher_model, self.target_layer, x_tensor, target_category=None)
        student_cam = compute_gradcam_heatmap(self.model, self.target_layer, x_tensor, target_category=None)

        # Normalize heatmaps
        teacher_cam = torch.from_numpy(teacher_cam).to(x_tensor.device)
        student_cam = torch.from_numpy(student_cam).to(x_tensor.device)

        teacher_cam = (teacher_cam - teacher_cam.min()) / (teacher_cam.max() - teacher_cam.min() + 1e-8)
        student_cam = (student_cam - student_cam.min()) / (student_cam.max() - student_cam.min() + 1e-8)

        # Compute the distillation loss between the heatmaps
        distillation_loss = F.mse_loss(student_cam, teacher_cam)

        # Combine the losses
        total_loss = student_loss + self.distillation_weight * distillation_loss
        return total_loss


def get_activation(model, target_layer, x):
    activations = []

    def hook(module, input, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(hook)
    _ = model(x)
    handle.remove()
    return activations[0]


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


if __name__ == '__main__':
    teacher_model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATIONS
    )
    teacher_model.to(DEVICE)

    # Define preprocessing function
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Create training and validation datasets
    train_dataset = SubstationDataset(x_train_dir, y_train_dir, augmentation=get_training_augmentation(),
                                      preprocessing=get_preprocessing(preprocessing_fn))

    val_dataset = SubstationDataset(x_val_dir, y_val_dir, augmentation=get_validation_augmentation(),
                                    preprocessing=get_preprocessing(preprocessing_fn))

    # Create data loaders
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Define loss function, metrics, and optimizer
    loss = DiceLoss(mode='multiclass')
    metrics = [IoU(threshold=0.5)]
    optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.0001)

    # Define training and validation epochs
    train_epoch = TrainEpoch(
        teacher_model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = ValidEpoch(
        teacher_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # Training loop
    max_score = 0
    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch}')
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        # Save best model
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(teacher_model.state_dict(), './best_teacher_model.pth')
            print('Best model saved!')

    # Load the trained teacher model
    teacher_model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATIONS
    )
    teacher_model.load_state_dict(torch.load('./best_teacher_model.pth'))
    teacher_model.to(DEVICE)
    teacher_model.eval()  # Set teacher model to evaluation mode

    # Initialize the student model
    student_model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATIONS
    )
    student_model.to(DEVICE)

    # Define the student's segmentation loss function
    student_loss_fn = DiceLoss(mode='multiclass')

    # Define the distillation loss function with GradCAM
    distillation_weight = 0.1  # Adjust as needed
    target_layer = student_model.decoder.block1  # Replace with your target layer

    distillation_loss_fn = DistillationLoss(
        student_loss_fn=student_loss_fn,
        model=student_model,
        teacher_model=teacher_model,
        target_layer=target_layer,
        distillation_weight=distillation_weight
    )

    # Define optimizer and metrics for the student model
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.0001)
    metrics = [IoU(threshold=0.5)]

    # Training loop for the student model
    max_score = 0
    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch}')
        student_model.train()
        train_loss = 0
        train_iou = 0

        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Get student predictions
            student_output = student_model(x_batch)

            # Get teacher predictions (with no gradient computation)
            with torch.no_grad():
                teacher_output = teacher_model(x_batch)

            # Compute the distillation loss (with GradCAM)
            loss = distillation_loss_fn(student_output, teacher_output, y_batch, x_batch)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            # Compute IoU for the batch
            y_pred = torch.argmax(student_output, dim=1)
            y_true = torch.argmax(y_batch, dim=1)
            iou = IoU()(y_pred, y_true)
            train_iou += iou.item()

        # Compute average loss and IoU
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation
        student_model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                # Get student predictions
                student_output = student_model(x_batch)

                # Compute the student's segmentation loss
                loss = student_loss_fn(student_output, y_batch)

                # Accumulate metrics
                val_loss += loss.item()
                y_pred = torch.argmax(student_output, dim=1)
                y_true = torch.argmax(y_batch, dim=1)
                iou = IoU()(y_pred, y_true)
                val_iou += iou.item()

        # Compute average validation loss and IoU
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(
            f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

        # Save best model
        if max_score < val_iou:
            max_score = val_iou
            torch.save(student_model.state_dict(), './best_student_model.pth')
            print('Best student model saved!')

    # Initialize the baseline student model ======================
    baseline_student_model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,  # You can set this to None if you want to train from scratch
        classes=len(CLASSES),
        activation=ACTIVATIONS
    )
    baseline_student_model.to(DEVICE)
    # Use the same segmentation loss function
    baseline_loss_fn = DiceLoss(mode='multiclass')
    optimizer = torch.optim.Adam(baseline_student_model.parameters(), lr=0.0001)
    metrics = [IoU(threshold=0.5)]
    # Training loop for the baseline student model
    max_score = 0
    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch}')
        baseline_student_model.train()
        train_loss = 0
        train_iou = 0

        for x_batch, y_batch in tqdm(train_loader):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Get predictions
            output = baseline_student_model(x_batch)

            # Compute loss
            loss = baseline_loss_fn(output, y_batch)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item()
            y_pred = torch.argmax(output, dim=1)
            y_true = torch.argmax(y_batch, dim=1)
            iou = IoU()(y_pred, y_true)
            train_iou += iou.item()

        # Compute average loss and IoU
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation
        baseline_student_model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader):
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                # Get predictions
                output = baseline_student_model(x_batch)

                # Compute loss
                loss = baseline_loss_fn(output, y_batch)

                # Accumulate metrics
                val_loss += loss.item()
                y_pred = torch.argmax(output, dim=1)
                y_true = torch.argmax(y_batch, dim=1)
                iou = IoU()(y_pred, y_true)
                val_iou += iou.item()

        # Compute average validation loss and IoU
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(
            f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}')

        # Save best model
        if max_score < val_iou:
            max_score = val_iou
            torch.save(baseline_student_model.state_dict(), './best_baseline_student_model.pth')
            print('Best baseline student model saved!')

    # Load the best models
    teacher_model.load_state_dict(torch.load('./best_teacher_model.pth'))
    student_model.load_state_dict(torch.load('./best_student_model.pth'))
    baseline_student_model.load_state_dict(torch.load('./best_baseline_student_model.pth'))

    # Set models to evaluation mode
    teacher_model.eval()
    student_model.eval()
    baseline_student_model.eval()

    # Create test dataset and dataloader
    test_dataset = SubstationDataset(
        x_test_dir,
        y_test_dir,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn)
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Evaluate the teacher model
    teacher_loss, teacher_iou = evaluate_model(teacher_model, test_loader)
    print(f'Teacher Model - Test Loss: {teacher_loss:.4f}, Test IoU: {teacher_iou:.4f}')

    # Evaluate the student model with distillation
    student_loss, student_iou = evaluate_model(student_model, test_loader)
    print(f'Student Model with Distillation - Test Loss: {student_loss:.4f}, Test IoU: {student_iou:.4f}')

    # Evaluate the baseline student model
    baseline_loss, baseline_iou = evaluate_model(baseline_student_model, test_loader)
    print(f'Baseline Student Model - Test Loss: {baseline_loss:.4f}, Test IoU: {baseline_iou:.4f}')


    def visualize_predictions(model, dataloader, num_samples=5):
        model.eval()
        with torch.no_grad():
            for i, (x_batch, y_batch) in enumerate(dataloader):
                if i >= num_samples:
                    break
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                # Get predictions
                output = model(x_batch)
                y_pred = torch.argmax(output, dim=1).cpu().numpy()
                y_true = torch.argmax(y_batch, dim=1).cpu().numpy()
                image = x_batch.cpu().numpy().transpose(0, 2, 3, 1)[0]

                # Plot results
                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title('Input Image')
                plt.axis('off')

                plt.subplot(1, 3, 2)
                plt.imshow(y_true[0], cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(y_pred[0], cmap='gray')
                plt.title('Predicted Mask')
                plt.axis('off')

                plt.show()


    # Visualize predictions for each model
    print('Teacher Model Predictions:')
    visualize_predictions(teacher_model, tqdm(test_loader))

    print('Student Model with Distillation Predictions:')
    visualize_predictions(student_model, tqdm(test_loader))

    print('Baseline Student Model Predictions:')
    visualize_predictions(baseline_student_model, tqdm(test_loader))
