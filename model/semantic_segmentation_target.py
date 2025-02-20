import torch
import numpy as np


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        # Convert the mask to a torch tensor if it's a numpy array
        if isinstance(mask, np.ndarray):
            self.mask = torch.from_numpy(mask).float()
        else:
            self.mask = mask.float()
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


class GroundTruthSegmentationTarget:
    def __init__(self, mask):
        self.mask = mask  # Ground truth mask tensor

    def __call__(self, model_output):
        # Assume model_output shape is (batch_size, num_classes, H, W)
        # We extract the output for all classes
        # The target is the dot product between the model's output and the ground truth mask
        return (model_output * self.mask).sum()
