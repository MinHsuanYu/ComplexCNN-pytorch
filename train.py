import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from models.alexNet import AlexNet1D

class ComplexDataset(Dataset):
    def __init__(self, data_dir):
        data_names = os.listdir(data_dir)
        self.input = []
        self.target = []
        for csv_name in data_names:
            csv_file = pd.read_csv(os.path.join(data_dir, csv_name))
            self.input.append(csv_file.loc[:, ["All_Real", "All_Imag"]].values)
            self.target.append(csv_file.loc[:, ["Cr_Real", "Cr_Imag"]].values)
        self.n_samples = len(self.input)
        self.input = np.array(self.input, dtype=np.float32)
        self.target = np.array(self.target, dtype=np.float32)
        self.input = torch.tensor(self.input).permute(0, 2, 1)
        self.target = torch.tensor(self.target).permute(0, 2, 1)
    def __getitem__(self, index):
        return self.input[index], self.target[index]
    def __len__(self):
        return self.n_samples

epochs = 1
batch_size = 20
Criterion = nn.MSELoss()
lr = 0.001
decay_step = 50
ethlon = 0.5


    
input_path = '0830/'
complex_dataset = ComplexDataset(input_path)
data_loader = DataLoader(dataset=complex_dataset, batch_size = batch_size, shuffle = True)
model = AlexNet1D(64, 7)
opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

loss_recorder = []

for epoch in range(epochs):
    print("="*10, "epoch: ", epoch, "="*10, sep='')
    for i, (inputs, targets) in enumerate(data_loader):
        opt.zero_grad()
        outputs = model(inputs)
        Loss = Criterion(outputs, targets)
        loss_recorder.append(Loss.item())
        print("Loss: ", Loss)
        Loss.backward()
        opt.step()

plt.plot(loss_recorder)
plt.show()