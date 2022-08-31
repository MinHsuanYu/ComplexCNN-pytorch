from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#Complex Conv 2 times
class ComplexCon2D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ComplexCon, self).__init__()
        self.Conv_real_1 = nn.Conv2d(in_channel//2, out_channel//2, 3, 1, 1, bias=False)
        self.Conv_image_1 = nn.Conv2d(in_channel//2, out_channel//2, 3, 1, 1, bias=False)
        self.Conv_real_2 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1, bias=False)
        self.Conv_image_2 = nn.Conv2d(out_channel//2, out_channel//2, 3, 1, 1, bias=False)
    def forward(self, x):
        print("x shape: ", x.shape)
        x_real, x_image = torch.chunk(x, 2, dim=1)
        print(x_real.shape, x_image.shape)
        x1_real = self.Conv_real_1(x_real) - self.Conv_image_1(x_image)
        print(x1_real.shape)
        x1_image = self.Conv_image_1(x_real) + self.Conv_real_1(x_image)
        print(x1_image.shape)
        x2_real = self.Conv_real_2(x1_real) - self.Conv_image_2(x1_image)
        print(x2_real.shape)
        x2_image = self.Conv_image_2(x1_real) + self.Conv_real_2(x1_image)
        print(x2_image.shape)
        return torch.cat((x2_real, x2_image), dim=1)

class ComplexCon1D(nn.Module):
    