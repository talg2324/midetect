import torch
import torch.nn as nn
import math

""" CNN for 4 sec signals @ 500Hz = 2000 samples"""
class CNN1D(nn.Module):

    def __init__(self, layerparams, batch_size, dropout, wantscuda, output_dim=1):
        super(CNN1D, self).__init__()

        self.batch_size = batch_size
        self.layerparams = layerparams
        self.output_dim = output_dim
        self.signal_len = 2000

        self.layers = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        reduction = self.signal_len

        # self.maxpools = [4, 9, 14]
        self.maxpools = [2, 5]
        no_layers = 0

        for p in layerparams:

            self.layers.append(nn.Conv1d(in_channels=p['in'], out_channels=p['out'], kernel_size=p['kernel'], stride=p['stride']))
            reduction = self.reduction(reduction, p['kernel'], p['stride'])
            no_layers += 1

            # self.layers.append(nn.BatchNorm1d(p['out']))
            # no_layers += 1

            # Max Pooling
            if no_layers in self.maxpools:
                self.layers.append(nn.MaxPool1d(kernel_size=10-no_layers, stride=1))
                reduction = self.reduction(reduction, 10-no_layers, 1)
                no_layers += 1



        # Output layers
        self.conv2dense = nn.Linear(in_features=layerparams[-1]['out']*reduction, out_features=64)
        self.dense = nn.Linear(in_features=64, out_features=32)
        self.preout = nn.Linear(32,1)

        self.out = nn.Sigmoid()

        # Reusable layers
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):

        x_i = x.transpose(1,2).float()

        for depth in range(len(self.layers)):

            x_i = self.layers[depth](x_i)

            if depth in self.maxpools:
                x_i = self.relu(x_i)
                x_i = self.dropout(x_i)

        # Flatten features to FC
        x_out = x_i.flatten(start_dim=1)
        x_out = self.conv2dense(x_out)
        x_out = self.relu(x_out)
        x_out = self.dropout(x_out)

        # One FC layer
        x_out = self.dense(x_out)
        x_out = self.relu(x_out)
        x_out = self.dropout(x_out)

        # Reduce and sigmoid output
        x_out = self.preout(x_out)
        x_out = self.dropout(x_out)
        x_out = self.out(x_out)

        return x_out.view(self.batch_size)

    def model_type(self):
        return "CNN"

    def reduction(self, length, filter_size, stride):
        return math.ceil((length-filter_size+1)/stride)

