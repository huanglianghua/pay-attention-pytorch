'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGG_ATT': [64, 64, 128, 128, 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M', 512, 'M', 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_ATT(nn.Module):
    def __init__(self, mode='pc'):
        super(VGG_ATT, self).__init__()
        self.mode = mode

        self.features = self._make_layers(cfg['VGG_ATT'])
        self.classifier = nn.Linear(512, 10)

        self.l1 = nn.Sequential(*list(self.features)[:22])
        self.l2 = nn.Sequential(*list(self.features)[22:32])
        self.l3 = nn.Sequential(*list(self.features)[32:42])

        if mode == 'pc':
            self.u1 = nn.Conv2d(256, 1, 1)
            self.u2 = nn.Conv2d(512, 1, 1)
            self.u3 = nn.Conv2d(512, 1, 1)

        self.conv_out = nn.Sequential(*list(self.features)[42:50])
        self.fc1 = nn.Linear(512, 512)

        self.fc1_l1 = nn.Linear(512, 256)
        self.fc1_l2 = nn.Linear(512, 512)
        self.fc1_l3 = nn.Linear(512, 512)

        self.fc2 = nn.Linear(256 + 512 + 512, 10)

    def forward(self, x):
        l1 = self.l1(x)
        l2 = self.l2(l1)
        l3 = self.l3(l2)

        conv_out = self.conv_out(l3)
        fc1 = self.fc1(conv_out.view(conv_out.size(0), -1))
        fc1_l1 = self.fc1_l1(fc1)
        fc1_l2 = self.fc1_l2(fc1)
        fc1_l3 = self.fc1_l3(fc1)

        att1 = self._compatibility_fn(l1, fc1_l1, level=1)
        att2 = self._compatibility_fn(l2, fc1_l2, level=2)
        att3 = self._compatibility_fn(l3, fc1_l3, level=3)

        g1 = self._weighted_combine(l1, att1)
        g2 = self._weighted_combine(l2, att2)
        g3 = self._weighted_combine(l3, att3)

        g = torch.cat((g1, g2, g3), dim=1)
        out = self.fc2(g)

        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _compatibility_fn(self, l, g, level):
        if self.mode == 'dp':
            att = l * g.unsqueeze(2).unsqueeze(3)
            att = att.sum(1).unsqueeze(1)

            size = att.size()
            att = att.view(att.size(0), att.size(1), -1)
            att = F.softmax(att, dim=2)
            att = att.view(size)
        elif self.mode == 'pc':
            att = l + g.unsqueeze(2).unsqueeze(3)

            if level == 1:
                u = self.u1
            elif level == 2:
                u = self.u2
            elif level == 3:
                u = self.u3
            att = u(att)
            
            size = att.size()
            att = att.view(att.size(0), att.size(1), -1)
            att = F.softmax(att, dim=2)
            att = att.view(size)

        return att
    
    def _weighted_combine(self, l, att_map):
        g = l * att_map

        return g.view(g.size(0), g.size(1), -1).sum(2)


# net = VGG_ATT()
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
