import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join


def main():
    directory = "/home/moshelaufer/PycharmProjects/VAE/data/"
    file_list = [join(directory, file) for file in listdir(directory) if join(directory, file).endswith('.npy')]
    t0 = np.arange(np.load(file_list[0]).shape[0])
    t1 = np.arange(np.load(file_list[1]).shape[0])
    t2 = np.arange(np.load(file_list[2]).shape[0])
    plt.plot(t0, np.load(file_list[0]), 'r')
    plt.plot(t1, np.load(file_list[1]), 'b')
    plt.plot(t2, np.load(file_list[2]), 'g')
    plt.show()


if __name__ == '__main__':
    main()