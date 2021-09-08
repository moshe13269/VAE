import torch.nn as nn
import torch
from torch.nn import functional as F


def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # torch.nn.init.normal_(m,0.0, 1)
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
        # m.bias.data.zero_()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.bnm3 = nn.BatchNorm2d(num_features=16, momentum=0.1)

        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        self.bnm6 = nn.BatchNorm2d(num_features=32, momentum=0.1)

        self.conv7 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.bnm9 = nn.BatchNorm2d(num_features=64, momentum=0.1)

        self.conv10 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv11 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bnm11 = nn.BatchNorm2d(num_features=128, momentum=0.1)

        self.conv12 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv13 = nn.Conv2d(256, 256, 3, padding=1, stride=2)

        self.bnm13 = nn.BatchNorm2d(num_features=256, momentum=0.1)

        self.conv14 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.conv15 = nn.Conv2d(512, 512, 4)
        self.fc1 = nn.Linear(512, 7)


    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bnm3(x)

        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.bnm6(x)

        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = self.bnm9(x)

        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = self.bnm11(x)

        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.bnm13(x)

        x = F.relu(self.conv14(x))
        x = F.relu(self.conv15(x))
        x = self.fc1(x.view(x.shape[0], -1))
        return x

# a = Encoder()
# # # print( sum(p.numel() for p in a.parameters() if p.requires_grad))
# # a.apply(normal_init)
# p="/home/moshelaufer/Documents/dataset/results10/model21.pt"
# a.load_state_dict(torch.load(p)['model_state_dict'])
# d = torch.rand(1,1,512,512).float()

# m = VAE().float()
# d = torch.rand(3,1,256,256).float()
# x = m(d)
# print(x[0].shape, x[1].shape)
# print(x[0].shape == d.shape)
