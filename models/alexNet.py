from re import fullmatch
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .complexConv import ComplexCon1D, ComplexCon2D

class AlexNet1D(nn.Module):
    def make_layers_with_pooling(self, in_channel, out_channel):
        return nn.Sequential(
            ComplexCon1D(in_channel, out_channel).to(self.device),
            nn.MaxPool1d(3, 2, 1)
        )
    def __init__(self, hidden_channel, layer_nums=9,  regression_channel=2048):
        super(AlexNet1D, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_layer = self.make_layers_with_pooling(2, hidden_channel)
        self.hidden_layers = []
        for i in range(layer_nums-2):    #except first & last
            self.hidden_layers.append(self.make_layers_with_pooling(hidden_channel, hidden_channel))
        self.feature_final = self.make_layers_with_pooling(hidden_channel, hidden_channel)
        self.Dense = nn.Linear(hidden_channel * (regression_channel // (2 ** layer_nums)), 2048)
        self.real = nn.Linear(2048, regression_channel)
        self.image = nn.Linear(2048, regression_channel)
        self.regression_channel = regression_channel

    def forward(self, x):
        hidden = self.in_layer(x)
        for layers in self.hidden_layers:
            hidden = layers(hidden)
        feature = self.feature_final(hidden)
        feature = feature.flatten(1)
        fully_connection = self.Dense(feature)
        out_real = self.real(fully_connection)
        out_iamge = self.image(fully_connection)
        out_real = out_real.reshape(-1, 1, self.regression_channel)
        out_iamge = out_iamge.reshape(-1, 1, self.regression_channel)
        outputs = torch.cat((out_real, out_iamge), dim=1)
        return outputs
        