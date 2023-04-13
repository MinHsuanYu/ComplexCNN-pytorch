import torch
from torch.utils.data import Dataset

import os
import pandas as pd


class MRIDataset(Dataset):
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