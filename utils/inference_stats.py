import pandas as pd
import torch
from models.model_VAE import VAE
from models.model_Encoder import Encoder
from utils.dataloader import Dataset
import numpy as np
import os
from utils.util import denormalized_vector, convert_label4output
import matplotlib.pyplot as plt


class Results:

    def __init__(self, path2save, path2encoder, path2vae, path2dataset, path2csv, vae=0):
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
                   '30.lfo1amount', '28.lfo1rate', '3.cufoff'] #, '4.resonance'
        df = pd.DataFrame(np_array, columns=columns)
        if self.vae:
            df.to_csv(os.path.join(self.path2save,'vae.csv'))
        else:
            df.to_csv(os.path.join(self.path2save, 'encoder.csv'))
        print('csv file had been saved')

    def predict_param(self):
        dataset = Dataset(self.path2dataset, self.path2csv, train=0)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        predicted_arr = np.empty([len(data_loader.dataset), 6], dtype=float)

        with torch.no_grad():
            c = 1
            d=e=f=0
            d1 = e1 = f1 = 0
            for batch_num, data in enumerate(data_loader):
                if batch_num % 10 == 0 and batch_num > 0:
                    print('sample num: {}'.format(batch_num))
                    break
                spec = data[0].float().to(self.device)
                label = data[1]
                if self.vae:
                    _, vector = self.VAE(spec)
                    print(vector)
                else:
                    vector = self.Encoder(spec)
                    print(vector)
                vector = vector.cpu().numpy() #* np.asarray([0.75, 0.75, 0.43, 1.0, 0.64, 1.0, 1.0])
                vector = vector.squeeze()
                vector = convert_label4output(vector)
                # print(vector)
                # label = label * np.asarray([0.75, 0.75, 0.43, 1.0, 0.64, 1.0, 1.0])
                vector = np.around(vector.numpy(), decimals=2)
                # label = np.around(label[0].numpy(), decimals=3)
                label = np.around(np.asarray(label), decimals=2)
                if vector[0]!=0.25:
                    d+=1
                if label[0]!=0.25:
                    d1+=1
                if vector[1]!=0.25:
                    e+=1
                if label[1]!=0.25:
                    e1+=1
                if vector[2]!=0.43:
                    f+=1
                if label[2]!=0.43:
                    f1+=1
                # print(vector)
                # print(label)
                print('\n')
                predicted_arr[c] = vector
                predicted_arr[c+1] = label
                c += 2
        print(d,e,f)
        print(d1, e1, f1)
        return predicted_arr


def main():
    path2dataset = ["/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0/",
                    "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly_res0.csv"]

    path2save = "/home/moshelaufer/PycharmProjects/VAE/results"
    path2encoder = "/home/moshelaufer/PycharmProjects/VAE/data2/model_encoder.pt"
    path2vae = "/home/moshelaufer/PycharmProjects/VAE/data2/modelVAE_KL2.pt"

    inference_model = Results(path2save, path2encoder, path2vae, path2dataset[0], path2dataset[1])
    inference_model.load_weight_model()

    predicted_arr = inference_model.predict_param()
    inference_model.save_results2csv(predicted_arr)
    print('Predicted process of VAE is done')


if __name__ == "__main__":
    main()
