import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'  # or 'cpu'