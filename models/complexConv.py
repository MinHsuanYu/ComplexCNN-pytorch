from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#Complex Conv 2 times
class ComplexCon2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ComplexCon2D, self).__init__()
        self.Conv_real_1 = nn.Conv2d(in_channel//2, out_channel//2, 3, 1, 1)
        self.Conv_image_1 = nn.Conv2d(in_channel//2, out_channel//2, 3, 1, 1)
        self.Conv_real_2 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1, bias=False)
        self.Conv_image_2 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1, bias=False)
    def forward(self, x):
        x_real, x_image = torch.chunk(x, 2, dim=1)
        x1_real = self.Conv_real_1(x_real) - self.Conv_image_1(x_image)
        x1_image = self.Conv_image_1(x_real) + self.Conv_real_1(x_image)
        x2_real = self.Conv_real_2(x1_real) - self.Conv_image_2(x1_image)
        x2_image = self.Conv_image_2(x1_real) + self.Conv_real_2(x1_image)
        return torch.cat((x2_real, x2_image), dim=1)

class ComplexCon1D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ComplexCon1D, self).__init__()
        self.Conv_real_1 = nn.Conv1d(in_channel//2, out_channel//2, 3, 1, 1)
        self.Conv_image_1 = nn.Conv1d(in_channel//2, out_channel//2, 3, 1, 1)
        self.Conv_real_2 = nn.Conv1d(out_channel//2, out_channel//2, 3, 1, 1, bias=False)
        self.Conv_image_2 = nn.Conv1d(out_channel//2, out_channel//2, 3, 1, 1, bias=False)
    def forward(self, x):
        x_real, x_image = torch.chunk(x, 2, dim=1)
        print(x_real.dtype)
        x1_real = self.Conv_real_1(x_real) - self.Conv_image_1(x_image)
        x1_image = self.Conv_image_1(x_real) + self.Conv_real_1(x_image)
        x2_real = self.Conv_real_2(x1_real) - self.Conv_image_2(x1_image)
        x2_image = self.Conv_image_2(x1_real) + self.Conv_real_2(x1_image)
        return torch.cat((x2_real, x2_image), dim=1)