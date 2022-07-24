"""
## ACMMM 2022
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.archs.CDC import cdcconv
from models.archs.arch_util import Refine



class ProcessBlock(nn.Module):
    def __init__(self, nc):
        super(ProcessBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1))
        self.cdc =cdcconv(nc,nc)
        self.fuse = nn.Conv2d(2*nc,nc,1,1,0)

    def forward(self, x):
        x_conv = self.conv(x)
        x_cdc = self.cdc(x)
        x_out = self.fuse(torch.cat([x_conv,x_cdc],1))

        return x_out



class DualBlock(nn.Module):
    def __init__(self, nc):
        super(DualBlock,self).__init__()
        self.relu = nn.ReLU()
        self.norm = nn.InstanceNorm2d(nc,affine=True)
        self.prcessblock = ProcessBlock(nc)
        self.fuse1 = nn.Conv2d(2*nc,nc,1,1,0)
        self.fuse2 = nn.Conv2d(2*nc,nc,1,1,0)
        self.post = nn.Sequential(nn.Conv2d(2*nc,nc,3,1,1),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv2d(nc,nc,3,1,1))

    def forward(self, x):
        x_norm = self.norm(x)
        x_p = self.relu(x)
        x_n = self.relu(-x)
        x_p = self.prcessblock(x_p)
        x_n = -self.prcessblock(x_n)
        x_p = self.fuse1(torch.cat([x_norm,x_p], 1))
        x_n = self.fuse2(torch.cat([x_norm,x_n], 1))
        x_out = self.post(torch.cat([x_p,x_n],1))

        return x_out+x



class DualProcess(nn.Module):
    def __init__(self, nc):
        super(DualProcess,self).__init__()
        self.conv1 = DualBlock(nc)
        self.conv2 = DualBlock(nc)
        self.conv3 = DualBlock(nc)
        self.conv4 = DualBlock(nc)
        self.conv5 = DualBlock(nc)
        self.cat = nn.Conv2d(5 * nc, nc, 1, 1, 0)
        self.refine = Refine(nc,3)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        xout = self.cat(torch.cat([x1,x2,x3,x4,x5],1))
        xfinal = self.refine(xout)

        return xfinal,xout



class InteractNet(nn.Module):
    def __init__(self, nc):
        super(InteractNet,self).__init__()
        self.extract = nn.Conv2d(3,nc,3,1,1)
        self.dualprocess = DualProcess(nc)

    def forward(self, x):
        x_pre = self.extract(x)
        x_final, xout = self.dualprocess(x_pre)

        return torch.clamp(x_final+0.00001,0.0,1.0),xout