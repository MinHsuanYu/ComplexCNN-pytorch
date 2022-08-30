from os import pread
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

real = torch.tensor([1, 2], dtype=torch.float32)
image = torch.tensor([3, 4], dtype=torch.float32)
tmp = torch.complex(real, image)
print(tmp)