from statistics import mode
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
        self.data_dir = data_dir
        self.data_names = os.listdir(data_dir)
        self.n_samples = len(self.data_names)
    def __getitem__(self, index):
        current_data_name = self.data_names[index]
        csv_file = pd.read_csv(os.path.join(self.data_dir, current_data_name))
        data_input = csv_file.loc[:, ["All_Real", "All_Imag"]].values.astype(np.float32)
        data_target =  csv_file.loc[:, ["Cr_Real", "Cr_Imag"]].values.astype(np.float32)
        data_input = torch.tensor(data_input).permute(1, 0)
        data_target = torch.tensor(data_target).permute(1, 0)
        return data_input, data_target
    def __len__(self):
        return self.n_samples


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100
batch_size = 20
Criterion = nn.MSELoss()
lr = 0.001
decay_step = 50
ethlon = 0.5


    
input_path = '0830/'
complex_dataset = ComplexDataset(input_path)
data_loader = DataLoader(dataset=complex_dataset, batch_size = batch_size, shuffle = True)
model = AlexNet1D(64, 7).to(device)
opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

loss_recorder = []

for epoch in range(epochs):
    print("="*10, "epoch: ", epoch, "="*10, sep='')
    for i, (inputs, targets) in enumerate(data_loader):
        opt.zero_grad()
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')
        outputs = model(inputs)
        Loss = Criterion(outputs, targets)
        loss_recorder.append(Loss.item())
        print("Loss: ", Loss)
        Loss.backward()
        opt.step()

torch.save(model, 'model.pt')
plt.plot(loss_recorder)
plt.show()
