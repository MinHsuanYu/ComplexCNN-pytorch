import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from complexConv import ComplexCon

class AlexNet(nn.Module):
    def __init__(self, in_channel):
        basis = nn.Sequential(
            ComplexCon(2, 64)
            nn.max
        )