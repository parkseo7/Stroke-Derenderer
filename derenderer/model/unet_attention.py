"""Modified UNet model to detect thin areas, namely skeletons. Uses an attention
layer and outputs different resolutions of the image.
All code is taken from https://github.com/namdvt/skeletonization

Later, we can remove some of the upper layers to get the resolution we want.
Ideally we want to output the max resolution, but for optimization purposes this
is an option.
"""

from collections import OrderedDict

import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable


class Conv2d(nn.Module):
    """Convolutional layer with batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True, dilation=1):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, bias=bias,
                              dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpConv2d(nn.Module):
    """Upsampling convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super(UpConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 
                                       stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True):
        super(DoubleConv2d, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding, bias=bias)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size, 
                            stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AttentionGroup(nn.Module):
    def __init__(self, num_channels):
        super(AttentionGroup, self).__init__()
        self.conv1 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv_1x1 = nn.Conv2d(num_channels, 3, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        s = torch.softmax(self.conv_1x1(x), dim=1)

        att = s[:,0,:,:].unsqueeze(1) * x1 \
            + s[:,1,:,:].unsqueeze(1) * x2 \
            + s[:,2,:,:].unsqueeze(1) * x3

        return x + att


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)



class Encoder(nn.Module):
    def __init__(self, in_channels=3, init_features=64):
        super(Encoder, self).__init__()

        # Use initial features as base:
        feats1 = init_features
        feats2 = init_features * 2
        feats3 = init_features * 4
        feats4 = init_features * 8
        feats5 = init_features * 16 # For 128 version, remove this.

        self.conv1 = DoubleConv2d(in_channels, feats1, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(feats1, feats2, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(feats2, feats3, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(feats3, feats4, kernel_size=3, padding=1)
        self.conv5 = DoubleConv2d(feats4, feats5, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(kernel_size=2)

        self.att1 = AttentionGroup(feats1)
        self.att2 = AttentionGroup(feats2)
        self.att3 = AttentionGroup(feats3)
        self.att4 = AttentionGroup(feats4)
        self.att5 = AttentionGroup(feats5)


    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.att1(out1)

        out2 = self.conv2(self.pooling(out1))
        out2 = self.att2(out2)

        out3 = self.conv3(self.pooling(out2))
        out3 = self.att3(out3)

        out4 = self.conv4(self.pooling(out3))
        out4 = self.att4(out4)

        out5 = self.conv5(self.pooling(out4))
        out5 = self.att5(out5)

        return out1, out2, out3, out4, out5


class Decoder(nn.Module):
    def __init__(self, out_channels=1, init_features=64):
        super(Decoder, self).__init__()
        feats1 = init_features
        feats2 = init_features * 2
        feats3 = init_features * 4
        feats4 = init_features * 8
        feats5 = init_features * 16

        self.upconv1 = UpConv2d(feats5, feats4, kernel_size=2, stride=2)
        self.upconv2 = UpConv2d(feats4, feats3, kernel_size=2, stride=2)
        self.upconv3 = UpConv2d(feats3, feats2, kernel_size=2, stride=2)
        self.upconv4 = UpConv2d(feats2, feats1, kernel_size=2, stride=2)

        self.conv1 = DoubleConv2d(feats5, feats4, kernel_size=3, padding=1)
        self.conv2 = DoubleConv2d(feats4, feats3, kernel_size=3, padding=1)
        self.conv3 = DoubleConv2d(feats3, feats2, kernel_size=3, padding=1)
        self.conv4 = DoubleConv2d(feats2, feats1, kernel_size=3, padding=1)

        self.conv1x1 = nn.Conv2d(feats1, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_conv_128 = nn.Conv2d(feats2, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_conv_64 = nn.Conv2d(feats3, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_conv_32 = nn.Conv2d(feats4, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.ca1 = ChannelAttention(feats4)
        self.sa1 = SpatialAttention()

        self.ca2 = ChannelAttention(feats3)
        self.sa2 = SpatialAttention()

        self.ca3 = ChannelAttention(feats2)
        self.sa3 = SpatialAttention()

        self.ca4 = ChannelAttention(feats1)
        self.sa4 = SpatialAttention()


    def forward(self, out1, out2, out3, out4, x):
        x = self.upconv1(x)
        x = torch.cat([x, out4], dim=1)
        x = self.conv1(x)
        x = self.ca1(x) * x
        x = self.sa1(x) * x
        aux_32 = self.aux_conv_32(x)

        x = self.upconv2(x)
        x = torch.cat([x, out3], dim=1)
        x = self.conv2(x)
        x = self.ca2(x) * x
        x = self.sa2(x) * x
        aux_64 = self.aux_conv_64(x)

        x = self.upconv3(x)
        x = torch.cat([x, out2], dim=1)
        x = self.conv3(x)
        x = self.ca3(x) * x
        x = self.sa3(x) * x
        aux_128 = self.aux_conv_128(x)

        x = self.upconv4(x)
        x = torch.cat([x, out1], dim=1)
        x = self.conv4(x)
        x = self.ca4(x) * x
        x = self.sa4(x) * x
        x = self.conv1x1(x)

        return x, aux_128, aux_64, aux_32


class UnetAttention(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UnetAttention, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, 
                               init_features=init_features)
        self.decoder = Decoder(out_channels=out_channels, 
                               init_features=init_features)

    def forward(self, x):
        out1, out2, out3, out4, x = self.encoder(x.float())
        x, aux_128, aux_64, aux_32 = self.decoder(out1, out2, out3, out4, x)

        return x.squeeze(), aux_128.squeeze(), aux_64.squeeze(), aux_32.squeeze()
    

class UnetAttentionEval(nn.Module):
    """Inference version, where the forward pass only outputs the highest
    resolution.
    """

    def __init__(self, in_channels=3, out_channels=1, init_features=64):
        super(UnetAttentionEval, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, 
                               init_features=init_features)
        self.decoder = Decoder(out_channels=out_channels, 
                               init_features=init_features)

    def forward(self, x):
        out1, out2, out3, out4, x = self.encoder(x.float())
        x, aux_128, aux_64, aux_32 = self.decoder(out1, out2, out3, out4, x)

        # B, C, H, W:
        return x
    

# LOSS FUNCTIONS:
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.01, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, preds, targets):
        BCE_loss = F.binary_cross_entropy(preds.view(-1), 
                                          targets.view(-1).float(), 
                                          reduction='none')
        targets = targets.type(torch.long)
        self.alpha = self.alpha.to(preds.device)
        
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        F_loss = F_loss.mean()

        if math.isnan(F_loss) or math.isinf(F_loss):
            F_loss = torch.zeros(1).to(preds.device)

        return F_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # outputs = torch.sigmoid(preds.squeeze())

        numerator = 2 * torch.sum(preds * targets) + self.smooth
        denominator = torch.sum(preds ** 2) + torch.sum(targets ** 2) + self.smooth
        soft_dice_loss = 1 - numerator / denominator

        return soft_dice_loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.alpha = 0.4
        self.dice_loss = DiceLoss()
        self.focal_loss = WeightedFocalLoss()
        self.w_dice = 1.
        self.w_focal = 100.
        self.S_dice = []
        self.S_focal = []


    def forward(self, preds, targets):
        preds = torch.sigmoid(preds.squeeze())
        
        dice_loss = self.dice_loss(preds, targets) * self.w_dice
        focal_loss = self.focal_loss(preds, targets) * self.w_focal

        return dice_loss, focal_loss