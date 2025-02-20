class XAIEnhancedDiceLoss(DiceLoss):
    def __init__(self, model, target_layer, category_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.target_layer = target_layer
        self.category_idx = category_idx
        self.__name__ = "XAIDiceLoss"
        self.alignment_weight = 0.00

    def forward(self, y_pred, y_true, x_tensor):
        # Ensure y_true is on the same device as y_pred
        y_true = y_true.to(y_pred.device)

        # Check for NaNs in inputs
        assert not torch.isnan(y_pred).any(), "y_pred contains NaNs"
        assert not torch.isnan(y_true).any(), "y_true contains NaNs"
        assert not torch.isnan(x_tensor).any(), "x_tensor contains NaNs"

        # Compute the standard Dice loss
        dice_loss = super().forward(y_pred, y_true)

        # Compute the Grad-CAM heatmap
        cam = HiResCAM(model=self.model, target_layers=[self.target_layer])
        y_pred_numpy = y_pred.detach().cpu().numpy()
        grayscale_cam = cam(input_tensor=x_tensor,
                            targets=[SemanticSegmentationTarget(self.category_idx, y_pred_numpy)])[0, :]

        # Resize the heatmap to the size of y_true
        grayscale_cam_resized = cv2.resize(grayscale_cam, (y_true.size(2), y_true.size(3)))

        # Convert to torch tensor
        grayscale_cam_tensor = torch.from_numpy(grayscale_cam_resized).float().to(y_pred.device).unsqueeze(0)

        # Normalize the heatmap
        grayscale_cam_tensor = (grayscale_cam_tensor - grayscale_cam_tensor.min()) / (
                grayscale_cam_tensor.max() - grayscale_cam_tensor.min() + 1e-8)  # Add epsilon to avoid division by zero

        # Ensure the dimensions match
        grayscale_cam_tensor = grayscale_cam_tensor.expand_as(y_true)

        # Align the heatmap with the ground truth
        alignment_loss = F.mse_loss(grayscale_cam_tensor, y_true.float())

        # Combine the Dice loss with the alignment loss
        total_loss = dice_loss + self.alignment_weight * alignment_loss
        print(f"Dice Loss: {dice_loss}, Alignment Loss: {alignment_loss}, Total Loss: {total_loss}")

        # Check for NaNs in the computed loss
        assert not torch.isnan(total_loss).any(), "total_loss contains NaNs"

        return total_loss


class WeightedDiceLoss(DiceLoss):
    def __init__(self, model, target_layer, category_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.target_layer = target_layer
        self.category_idx = category_idx
        self.__name__ = "WeightedDiceLoss"

    def forward(self, y_pred, y_true, x_tensor):
        y_true = y_true.to(y_pred.device)

        # Compute the Grad-CAM heatmaps
        cam = GradCAM(model=self.model, target_layers=[self.target_layer])
        y_pred_numpy = y_pred.detach().cpu().numpy()

        # Create targets per sample
        targets = [SemanticSegmentationTarget(self.category_idx, y_pred_numpy[i]) for i in range(y_pred_numpy.shape[0])]
        grayscale_cams = cam(input_tensor=x_tensor, targets=targets)

        # Resize and normalize heatmaps per sample
        weight_maps = []
        for i in range(grayscale_cams.shape[0]):
            cam_resized = cv2.resize(grayscale_cams[i], (y_true.size(3), y_true.size(2)))  # width, height
            # Normalize the heatmap to [0, 1]
            cam_normalized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
            weight_maps.append(cam_normalized)

        weight_maps = np.stack(weight_maps)  # Shape: (batch_size, H, W)
        weight_maps_tensor = torch.from_numpy(weight_maps).float().to(y_pred.device).unsqueeze(1)  # (batch_size,1,H,W)

        # Ensure y_pred and y_true are float tensors
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Compute weighted Dice loss
        intersection = (weight_maps_tensor * y_pred * y_true).sum(dim=(2, 3))
        union = (weight_maps_tensor * y_pred + weight_maps_tensor * y_true).sum(dim=(2, 3))
        dice_score = (2. * intersection + self.eps) / (union + self.eps)
        weighted_dice_loss = 1 - dice_score.mean()

        return weighted_dice_loss