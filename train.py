import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchsummary
import numpy as np

from alexNet import AlexNet1D

device = "cpu"


model = AlexNet1D(64,layer_nums = 7, regression_channel=2048)
x = torch.tensor(np.ndarray((1, 2, 2048))).to(device).float()
real, image = model(x)
print(real.shape, image.shape)


