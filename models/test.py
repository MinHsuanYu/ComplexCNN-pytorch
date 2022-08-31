import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchsummary

import complexConv

model = complexConv.ComplexCon(2, 16).cuda()
print(model)
torchsummary.summary(model, (2, 1, 2048))


