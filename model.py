import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import time

class CodeEnhancement(nn.Module):
    def __init__(self, num_points=2048):
        super(CodeEnhancement, self).__init__()
        self.conv1 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        x = self.leaky_relu(self.conv1(input.transpose(1,2)))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.leaky_relu(self.conv5(x))

        return x.transpose(1,2)


class WarpingGAN(nn.Module):
    def __init__(self, num_points=2048, m=128, dimofgrid=3):
        super(WarpingGAN, self).__init__()
        self.n = num_points  # input point cloud size.
        self.numgrid = int(2048 / m)
        self.dimofgrid = dimofgrid
        self.m = m  # 16 * 16.
        self.meshgrid = [[-0.2, 0.1, 4], [-0.2, 0.1, 4], [-0.2, 0.1, 8]]
        self.codegenerator = CodeEnhancement(num_points=num_points)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(int(512/self.numgrid)+self.dimofgrid+512, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 3, 1),
            # nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(int(512/self.numgrid)+3+512, 256, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(256, 64, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 3, 1),
            # nn.ReLU(),
        )

    def build_grid(self, batch_size):
        
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        z = np.linspace(*self.meshgrid[2])
        grid = np.array(list(itertools.product(x, y, z)))
        grid = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)
        grid = torch.tensor(grid)
        # print(grid.shape)
        return grid.float()

    def forward(self, input):
        input = self.codegenerator(input)
        input = input.transpose(1, 2).repeat(1, 1, self.m)  # [bs, 512, m]
        
        batch_size = input.size(0)
        splitinput = input.view(batch_size, self.numgrid, int(512/self.numgrid), self.m)
        globalfeature = input.view(batch_size, 1, 512, self.m).repeat(1,self.numgrid,1,1)
        gridlist = []
        for i in range(self.numgrid):
            gridlist.append(self.build_grid(input.shape[0]).transpose(1, 2).view(batch_size, 1, self.dimofgrid, self.m))
        
        if torch.cuda.is_available():
            for i in range(self.numgrid):
                gridlist[i] = gridlist[i].cuda()    
        
        grid = gridlist[0]
        for i in range(1,self.numgrid): 
            grid = torch.cat((grid,gridlist[i]),axis=1)
        concate1 = torch.cat((splitinput, globalfeature, grid), dim=2).transpose(1,2).reshape(batch_size, int(512/self.numgrid)+self.dimofgrid+512, 2048)  # [bs, 514, m]
        after_folding1 = self.mlp1(concate1)  # [bs, 3, m]
        concate2 = torch.cat((splitinput, globalfeature, after_folding1.reshape(batch_size,3,self.numgrid,self.m).transpose(1,2)), dim=2).transpose(1,2).reshape(batch_size, int(512/self.numgrid)+3+512, 2048)  # [bs, 515, m]
        after_folding2 = self.mlp2(concate2)  
        return after_folding2.transpose(1, 2)   # [bs, m ,3]


class Discriminator(nn.Module):
    def __init__(self, batch_size, features, num_points=2048):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        super(Discriminator, self).__init__()
        self.numpoints = num_points
        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
                                         nn.Linear(features[-1], features[-2]),
                                         nn.Linear(features[-2], features[-2]),
                                         nn.Linear(features[-2], 1))
        self.maxpool = nn.MaxPool1d(kernel_size=self.numpoints, return_indices=True)

    def forward(self, f):
        feat = f.transpose(1,2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out, indices = self.maxpool(feat)
        out = self.final_layer(out.squeeze(-1)) # (B, 1)

        return out, indices 

