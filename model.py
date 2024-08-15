import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from resnet import resnet50
from utils import *
from einops.einops import rearrange
from mmengine.model import BaseModule


class Head(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.gnn = GNN(self.in_channels, self.num_classes, neighbor_num=neighbor_num, metric=metric)
        self.sc = nn.Parameter(torch.FloatTensor(torch.zeros(self.num_classes, self.in_channels)))
        self.relu = nn.ReLU()
        nn.init.xavier_uniform_(self.sc)

        class_generate_layers = []
        for j in range(self.num_classes):
            p_layer = nn.AdaptiveAvgPool2d(1)
            class_generate_layers += [p_layer]

        self.class_generates = nn.ModuleList(class_generate_layers)
        self.class_linear = LinearBlock(self.in_channels, self.in_channels)


        self.mafg = MAFG(self.num_classes, self.in_channels)
        self.ccfg = CCFG(self.num_classes, self.in_channels, self.in_channels)
        self.post_concat = LinearBlock(self.in_channels * 2, self.in_channels)

    def forward(self, x):
        # MAFG
        branch_1 = unsqz(x)
        v_mafg = self.mafg(branch_1)

        # GNN
        v_gnn = self.gnn(v_mafg)

        # CCFG
        branch_2 = unsqz(x)
        v_ccfg = self.ccfg(branch_2)

        # Fusion
        f_fusion = torch.cat((v_gnn, v_ccfg), dim=2)

        # Post process
        f_fusion = self.post_concat(f_fusion)
        b, n, c = f_fusion.shape
        sc = self.sc
        sc = self.relu(sc)
        sc = F.normalize(sc, p=2, dim=-1)
        cl = F.normalize(f_fusion, p=2, dim=-1)
        cl = (cl * sc.view(1, n, c)).sum(dim=-1)
        return cl


class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, neighbor_num=4, metric='dots'):
        super(GNN, self).__init__()
        # neighbor_num: K in paper and we select the top-K nearest neighbors for each node feature.

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.metric = metric
        self.neighbor_num = neighbor_num

        # network
        self.U = nn.Linear(self.in_channels, self.in_channels)
        self.V = nn.Linear(self.in_channels, self.in_channels)
        self.bnv = nn.BatchNorm1d(num_classes)

        # init
        self.U.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.V.weight.data.normal_(0, math.sqrt(2. / self.in_channels))
        self.bnv.weight.data.fill_(1)
        self.bnv.bias.data.zero_()

    def forward(self, x):
        b, n, c = x.shape

        # build dynamical graph
        if self.metric == 'dots':
            si = x.detach()
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()

        elif self.metric == 'cosine':
            si = x.detach()
            si = F.normalize(si, p=2, dim=-1)
            si = torch.einsum('b i j , b j k -> b i k', si, si.transpose(1, 2))
            threshold = si.topk(k=self.neighbor_num, dim=-1, largest=True)[0][:, :, -1].view(b, n, 1)
            adj = (si >= threshold).float()


        else:
            raise Exception("Error: wrong metric: ", self.metric)

        # GNN process
        A = normalize_digraph(adj)
        aggregate = torch.einsum('b i j, b j k->b i k', A, self.V(x))
        x = self.relu(x + self.bnv(aggregate + self.U(x)))
        return x


def normalize_digraph(A):
    b, n, _ = A.shape
    node_degrees = A.detach().sum(dim=-1)
    degs_inv_sqrt = node_degrees ** -0.5
    norm_degs_matrix = torch.eye(n)
    dev = A.get_device()
    if dev >= 0:
        norm_degs_matrix = norm_degs_matrix.to(dev)
    norm_degs_matrix = norm_degs_matrix.view(1, n, n) * degs_inv_sqrt.view(b, n, 1)
    norm_A = torch.bmm(torch.bmm(norm_degs_matrix, A), norm_degs_matrix)
    return norm_A


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)
        self.fc.weight.data.normal_(0, math.sqrt(2. / out_features))
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, x):
        x = self.drop(x)
        x = self.fc(x).permute(0, 2, 1)
        x = self.relu(self.bn(x)).permute(0, 2, 1)

        return x


# MAFG And CCFG enhance GNN
class MACG(nn.Module):
    def __init__(self, num_classes=8, backbone='resnet50', neighbor_num=4, metric='dots'):
        super(MACG, self).__init__()
        if 'resnet' in backbone:
            if backbone == 'resnet50':
                self.backbone = resnet50()
            self.in_channels = self.backbone.fc.weight.shape[1]
            self.out_channels = self.in_channels // 4
            self.backbone.fc = None
        else:
            raise Exception("Error: wrong backbone name: ", backbone)

        self.global_linear = LinearBlock(self.in_channels, self.out_channels)
        self.head = Head(self.out_channels, num_classes, neighbor_num, metric)

    def forward(self, x):
        # x: b d c
        x = self.backbone(x)
        x = self.global_linear(x)
        cl = self.head(x)

        return cl


class MAFG(BaseModule):

    def __init__(self,
                 num_classes,
                 channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.num_classes = num_classes
        class_linear_layers = []
        for i in range(self.num_classes):
            layer = LinearBlock(channels, channels)
            class_linear_layers += [layer]
        self.class_linears = nn.ModuleList(class_linear_layers)
        self.conv0 = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels))
        self.conv3 = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        """Forward function."""
        u = x.clone()
        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        # representation generate
        x_mafg = []
        for i, layer in enumerate(self.class_linears):
            x_mafg.append(layer(sqz(x)).unsqueeze(1))
        x_mafg = torch.cat(x_mafg, dim=1)
        x_mafg = x_mafg.mean(dim=-2)

        return x_mafg


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


class CCFG(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CCFG, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
        class_generate_layers = []
        for j in range(self.num_classes):
            p_layer = nn.AdaptiveAvgPool2d(1)
            class_generate_layers += [p_layer]

        self.class_generates = nn.ModuleList(class_generate_layers)
        self.class_linear = LinearBlock(self.in_channels, self.in_channels)

    def forward(self, x, alpha=1):

        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        x_ccfg = []
        for j, layer in enumerate(self.class_generates):
            pool_result = layer(x1)
            pool_result = self.class_linear(sqz(pool_result))
            x_ccfg.append(pool_result)

        x_ccfg = torch.cat(x_ccfg, dim=1)

        return x_ccfg


if __name__ == "__main__":
    model = MACG()
    input = torch.randn(3, 3, 224, 224)
    output = model(input)
    print("output_shape:", output.shape)
