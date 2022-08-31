import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchsummary
import numpy as np

import complexConv

device = "cpu"


model = complexConv.ComplexCon1D(2, 16).double()
x = torch.tensor(np.ndarray((1, 2, 2048))).to(device)
y = model(x)
print(y)
