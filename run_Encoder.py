import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.model_Encoder import Encoder
from dataloader import Dataset
import time


def run_encoder():
    torch.cuda.empty_cache()
    file = open("/home/moshelaufer/PycharmProjects/VAE/data/process_state_encoder.txt", "a")
    device = torch.device('cuda:3')
    model = Encoder().to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.train()

    mse_criterion = nn.MSELoss().to(device)
    n_epochs = 100
    loss_arr_mid = []
    loss_arr_out = []

    print('start epoch')
    file.write('start epoch\n')
    batch_size = 150

    for epoch in range(n_epochs):
        dataset = Dataset("/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly/",
                          "/home/moshelaufer/Documents/TalNoise/TAL31.07.2021/20210727_data_150k_constADSR_CATonly.csv")
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                                  pin_memory=True, drop_last=True)
        print(len(data_loader.dataset))

        num_batch = len(data_loader.dataset) // batch_size
        loss_mid_tot = 0.0
        loss_out_tot = 0.0
        start_time = time.time()

        c1 = 0
        for batch_num, data in enumerate(data_loader):
            if batch_num % 200 == 0:
                print("sum samples = {} ".format(batch_num * batch_size))
            spec = data[0].float()
            label = data[1].float()
            spec = spec.to(device)
            label = label.to(device)
            re_spec, vector = model(spec)

            c1 += 1
            loss = mse_criterion(vector, label)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            loss_mid_tot += loss.item()

            if batch_num % 100 == 0 and batch_num > 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Mid loss: %f] Encoder"
                    % (epoch, n_epochs, batch_num, num_batch, loss_mid_tot / c1)
                )

        loss_mid_tot = loss_mid_tot / c1
        loss_arr_mid.append(loss_mid_tot)
        print("--- %s seconds ---" % (time.time() - start_time))
        file.write("--- %s seconds ---" % (time.time() - start_time))

        file.write('\n')
        file.write('loss_mid_tot = %f , loss_out_tot = %f  Encoder\n' % (loss_mid_tot, loss_out_tot))

        print('\n')

        file.write("Loss mid= {}, epoch = {} wl".format(loss_mid_tot, epoch))
        print("Loss mid train = {}, epoch = {}, batch_size = {} wl".format(loss_mid_tot, epoch, batch_size))
        outfile_epoch = "/home/moshelaufer/PycharmProjects/VAE/data/loss_arr_encoder.npy"
        np.save(outfile_epoch, np.asarray(loss_arr_mid))

        if epoch <= 2:
            path = "/home/moshelaufer/PycharmProjects/VAE/data/model_encoder.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model_optimizer.state_dict()}, path)
            print("Model had been saved")
        elif min(loss_arr_mid[:len(loss_arr_out) - 2]) >= loss_arr_mid[len(loss_arr_out) - 1]:
            path = "/home/moshelaufer/PycharmProjects/VAE/data/model_encoder.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model_optimizer.state_dict()}, path)
            print("Model had been saved")

    print("Training is over")
    file.write("Training is over\n")
    torch.no_grad()
    print("Weight file had successfully saved!!\n")
    file.close()
