import pandas as pd
import torch
from models.model_VAE import VAE
from models.model_Encoder import Encoder
from utils.dataloader import Dataset
import numpy as np
import os
from utils.util import denormalized_vector
import matplotlib.pyplot as plt


class Results:

    def __init__(self, path2save, path2encoder, path2vae, path2dataset, path2csv, vae=1):
        self.path2save = path2save
        self.device = torch.device('cuda:3')
        self.path2Encoder = path2encoder
        self.path2VAE = path2vae
        self.Encoder = Encoder().float().to(self.device)
        self.VAE = VAE().float().to(self.device)
        self.path2csv = path2csv
        self.path2dataset = path2dataset
        self.vae = vae

    def load_weight_model(self):
        self.Encoder.load_state_dict(torch.load(self.path2Encoder)['model_state_dict'])
        self.Encoder.eval()
        self.VAE.load_state_dict(torch.load(self.path2VAE)['model_state_dict'])
        self.VAE.eval()

    def save_results2csv(self, np_array):
        columns = ['24.osc2waveform', '26.lfo1waveform', '32.lfo1destination',
                   '30.lfo1amount', '28.lfo1rate', '3.cufoff', '4.resonance']
        df = pd.DataFrame(np_array, columns=columns)
        if self.vae:
            df.to_csv(os.path.join(self.path2save,'vae.csv'))
        else:
            df.to_csv(os.path.join(self.path2save, 'encoder.csv'))
        print('csv file had been saved')

    def predict_param(self):
        dataset = Dataset(self.path2dataset, self.path2dataset, train=0)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        predicted_arr = np.empty([len(data_loader.dataset), 7], dtype=float)

        with torch.no_grad():
            for batch_num, data in enumerate(data_loader):
                if batch_num % 1000 == 0 and batch_num > 0:
                    print('sample num: {}'.format(batch_num))
                    break
                spec = data[0].float().to(self.device)
                if VAE:
                    _, vector = self.VAE(spec)
                    print(vector)
                else:
                    vector = self.Encoder(spec)
                vector = vector / np.asarray([0.75, 0.75, 0.43, 1.0, 0.64, 1.0, 1.0])
                vector = denormalized_vector(vector)
                predicted_arr[batch_num] = vector
        return predicted_arr


def main():
    path2dataset = ["/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly/",
                    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly.csv"]

    path2save = "/home/moshelaufer/PycharmProjects/VAE/data"
    path2encoder = "/home/moshelaufer/PycharmProjects/VAE/data/model_encoder.pt"
    path2vae = "/home/moshelaufer/PycharmProjects/VAE/data/modelVAE_KL2.pt"

    inference_model = Results(path2save, path2encoder, path2vae, path2dataset[0], path2dataset[1])
    inference_model.load_weight_model()

    predicted_arr = inference_model.predict_param()
    inference_model.save_results2csv(predicted_arr)
    print('Predicted process of VAE is done')


if __name__ == "__main__":
    main()
