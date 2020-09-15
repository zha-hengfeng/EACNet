# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # classify
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# class DownSamplingBlock(nn.Module):
#     def __init__(self, nIn, nOut):
#         super().__init__()
#         self.nIn = nIn
#         self.nOut = nOut
#
#         if self.nIn < self.nOut:
#             nConv = nOut - nIn
#         else:
#             nConv = nOut
#
#         self.conv3x3 = nn.Conv2d(nIn, nOut, kernel_size=3, stride=2, padding=1)
#         #self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
#         self.max_pool = nn.MaxPool2d(2, stride=2)
#         self.bn_prelu = BNPReLU(nOut)
#
#     def forward(self, input):
#         output = self.conv3x3(input)
#
#         if self.nIn < self.nOut:
#             max_pool = self.max_pool(input)
#             output = torch.cat([output, max_pool], 1)
#
#         output = self.bn_prelu(output)
#
#         return output


class EAC_Module(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EAC_Module, self).__init__()
        self.chanel_in = in_dim
        self.max_pool_1 = nn.MaxPool2d(2, stride=2)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        pool = self.max_pool_1(x)
        pool_batch, pool_C, pool_h, pool_w = pool.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(pool).view(m_batchsize, -1, pool_h*pool_w)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(pool).view(m_batchsize, -1, pool_h*pool_w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


# EACNet_ERF
class EACNet_ERFv2(nn.Module):
    def __init__(self, num_classes):  # use encoder to pass pretrained encoder
        super().__init__()
        self.encoder = Encoder(num_classes)
        self.eac = EAC_Module(num_classes, num_classes)

    def forward(self, input):
        output = self.encoder(input)
        output = self.eac(output)
        output = F.interpolate(output, input.size()[2:], mode='bilinear', align_corners=False)
        return output

