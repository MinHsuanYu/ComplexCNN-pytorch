import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AlexNet(nn.Module):
    def __init__(self, in_channel, ):