from torch.utils.data import Dataset
import torch
from os import listdir
from os.path import isfile, join
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pickle
import pandas as pd


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


class Dataset(Dataset):
    def __init__(self, path2data, path2csv, train=1):
        self.train = train
        if train:
            self.path_list = [join(path2data, file) for file in listdir(path2data)
                              if int(file.replace('.wav', '')) % 7 != 0]
        else:
            self.path_list = [join(path2data, file) for file in listdir(path2data)
                              if int(file.replace('.wav', '')) % 7 == 0]
        self.csv_df = pd.read_csv(path2csv, skipinitialspace=True)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, index):

        try:
            fs, x = wavfile.read(self.path_list[index])
        except UnboundLocalError:
            print(self.path_list[index])
        f, t, Zxx = signal.stft(x, fs, nperseg=128, nfft=511, window='hamming')
        # f, t, Zxx = signal.stft(x, fs, nperseg=64, nfft=1023, window='hamming')
        Zxx = np.log(np.abs(Zxx) + 10 ** -10)
        Zxx = torch.tensor(Zxx[:, :256])
        # Zxx = ((Zxx - Zxx.mean()) / Zxx.std())
        # z_min = Zxx.min()
        # z_max = Zxx.max()
        # diff = z_max - z_min
        # Zxx = (Zxx - z_min) / diff
        Zxx = Zxx.unsqueeze(0)
        if np.sum(np.where(np.isnan(Zxx) == True, 1, 0)) > 0:
            print('index = {} is nan'.format(index))
        label = list(self.csv_df.loc[index])[1:]
        label = np.asarray(label) / np.asarray([0.75, 0.75, 0.43, 1.0, 0.64, 1.0, 1.0])
        # if self.train:
        #     label = convert_label2classes(label)
        #     # label = np.asarray(label)/np.asarray([0.75,0.75,0.43,1,0.64,1,1])
        #     label1 = torch.tensor(label[:3], dtype=torch.long)
        #     label2 = torch.tensor(label[3:], dtype=torch.float)
        #     label = torch.cat((label1, label2), dim=0)
        return Zxx, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = Dataset("/home/moshelaufer/Documents/TalNoise/TAL14.07.2021/sounds_constADSR/",
                          "/home/moshelaufer/Documents/TalNoise/TAL14.07.2021/20210713_data_150k_constADSR.csv")
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    c = 0
    cur_max = 0
    arr_max = []
    for i in range(len(data_loader)):

        a=next(iter(data_loader))
        arr_max.append(a[0].max().item())
        cur_max = max(cur_max,a[0].max().item())

