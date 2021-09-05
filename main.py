# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 10:28:14 2021

@author: moshelaufer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from model_VAE import VAE
from dataloader import Dataset
import time
import torch.nn.functional as F


def main():
    torch.cuda.empty_cache()
    file = open("../data/process_state_VAE_KL.txt", "a")
    device = torch.device('cuda:2')
    model = VAE().to(device)
    model_optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    model.train()

    mse_criterion = nn.MSELoss().to(device)
    # criterion_out = nn.KLDivLoss(reduction='batchmean').to(device)
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
            spec = data[0]
            label = data[1]
            spec = spec.to(device)
            label = label.to(device)
            re_spec, vector = model(spec)

            c1 += 1
            loss_m = mse_criterion(vector, label)
            loss_o = mse_criterion(spec, re_spec)
            loss = loss_o + loss_m
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            loss_mid_tot += loss.item()
            loss_out_tot += loss.item()

            if batch_num % 100 == 0 and batch_num > 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [Mid loss: %f] [Out loss: %f] VAE"
                    % (epoch, n_epochs, batch_num, num_batch, loss_mid_tot / c1, loss_out_tot / c1)
                )

        loss_mid_tot = loss_mid_tot / c1
        loss_out_tot = loss_out_tot / c1
        loss_arr_mid.append(loss_mid_tot)
        loss_arr_out.append(loss_out_tot)
        print("--- %s seconds ---" % (time.time() - start_time))
        file.write("--- %s seconds ---" % (time.time() - start_time))

        file.write('\n')
        file.write('loss_mid_tot = %f , loss_out_tot = %f  VAE\n' % (loss_mid_tot, loss_out_tot))

        print('\n')

        file.write("Loss mid= {}, epoch = {} wl".format(loss_mid_tot, epoch))
        file.write("Loss out= {}, epoch = {} wl".format(loss_out_tot, epoch))
        print("Loss mid train = {}, epoch = {}, batch_size = {} wl".format(loss_mid_tot, epoch, batch_size))
        print("Loss out train = {}, epoch = {}, batch_size = {} wl".format(loss_out_tot, epoch, batch_size))
        outfile_epoch = "data/loss_arr_mid2_KL2.npy"
        np.save(outfile_epoch, np.asarray(loss_arr_mid))
        outfile_epoch = "data/loss_arr_out2_KL2.npy"
        np.save(outfile_epoch, np.asarray(loss_arr_out))

        if epoch <= 2:
            path = "data/modelVAE_KL2.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model_optimizer.state_dict()}, path)
            print("Model had been saved")
        elif min(loss_arr_mid[:len(loss_arr_out) - 2]) >= loss_arr_mid[len(loss_arr_out) - 1]:
            path = "data/modelVAE_KL2.pt"
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model_optimizer.state_dict()}, path)
            print("Model had been saved")

    print("Training is over")
    file.write("Training is over\n")
    torch.no_grad()
    print("Weight file had successfully saved!!\n")
    file.close()


if __name__ == "__main__":
    main()
