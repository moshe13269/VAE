import torch.nn as nn
import torch
from torch.nn import functional as F


def normal_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        # torch.nn.init.normal_(m,0.0, 1)
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.fill_(0)
        # m.bias.data.zero_()


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.ups1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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
        self.fc1 = nn.Linear(512, 13)

        # decoder
        self.fc2 = nn.Linear(13, 512)
        self.conv20 = nn.ConvTranspose2d(512, 512, 4)
        self.conv21 = nn.ConvTranspose2d(512, 256, 3, padding=1)

        self.conv22 = nn.ConvTranspose2d(256, 256, 3, padding=1)
        self.conv23 = nn.ConvTranspose2d(256, 128, 3, padding=1)

        self.conv24 = nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.conv25 = nn.ConvTranspose2d(128, 64, 3, padding=1)

        self.conv26 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.conv27 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.conv28 = nn.ConvTranspose2d(64, 32, 3, padding=1)

        self.conv29 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.conv30 = nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.conv31 = nn.ConvTranspose2d(32, 16, 3, padding=1)

        self.conv32 = nn.ConvTranspose2d(16, 16, 3, padding=1, stride=2)
        self.conv33 = nn.ConvTranspose2d(16, 16, 3, padding=1)
        self.conv34 = nn.ConvTranspose2d(16, 16, 3, padding=1)
        self.conv35 = nn.ConvTranspose2d(16, 1, 4, padding=1)

        self.bnm21 = nn.BatchNorm2d(num_features=512, momentum=0.1)

        self.lrelu = nn.LeakyReLU(0.2, True)

    def weight_init(self):
        for m in self._modules:
            normal_init(self._modules[m])

    def forward(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.bnm3(x)
        layer1 = x

        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.bnm6(x)
        layer2 = x

        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = self.bnm9(x)
        layer3 = x

        x = self.lrelu(self.conv10(x))
        x = self.lrelu(self.conv11(x))
        x = self.bnm11(x)
        layer4 = x

        x = self.lrelu(self.conv12(x))
        x = self.lrelu(self.conv13(x))
        x = self.bnm13(x)
        layer5 = x

        x = self.lrelu(self.conv14(x))
        x = self.lrelu(self.conv15(x))
        out1 = self.fc1(x.view(x.shape[0], -1))

        x = self.lrelu(self.fc2(out1))
        x = x.unsqueeze(2).unsqueeze(3)

        x = self.lrelu(self.conv20(x))
        x = self.bnm21(x)
        x = self.lrelu(self.conv21(x))
        x = self.lrelu(self.ups1(x))

        x = self.lrelu(self.conv22(x + layer5))
        x = self.bnm13(x)
        x = self.lrelu(self.conv23(x))
        x = self.lrelu(self.ups1(x))

        x = self.lrelu(self.conv24(x + layer4))
        x = self.bnm11(x)
        x = self.lrelu(self.conv25(x))
        x = self.lrelu(self.ups1(x))

        x = self.lrelu(self.conv26(x + layer3))
        x = self.bnm9(x)
        x = self.lrelu(self.conv27(x))
        x = self.lrelu(self.conv28(x))
        x = self.lrelu(self.ups1(x))

        x = self.lrelu(self.conv29(x + layer2))
        x = self.bnm6(x)
        x = self.lrelu(self.conv30(x))
        x = self.lrelu(self.conv31(x))
        x = self.lrelu(self.ups1(x))

        x = self.lrelu(self.conv32(x + layer1))
        x = self.bnm3(x)
        x = self.lrelu(self.conv33(x))
        x = self.lrelu(self.conv34(x))
        x = self.lrelu(self.conv35(x))

        return x, out1

# a = VAE()
# # # print( sum(p.numel() for p in a.parameters() if p.requires_grad))
# # a.apply(normal_init)
# p="/home/moshelaufer/Documents/dataset/results10/model21.pt"
# a.load_state_dict(torch.load(p)['model_state_dict'])
# d = torch.rand(1,1,512,512).float()

# m = VAE().float()
# d = torch.rand(10,1,256,256).float()
# x = m(d)
# print(x[0].shape, x[1].shape)
# print(x[0].shape == d.shape)
