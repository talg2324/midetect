import torch
import torch.nn as nn

# class CNN1D(nn.Module):

#     def __init__(self, input_dim, batch_size, dropout, wantscuda, output_dim=1):
#         super(CNN1D, self).__init__()

#         self.input_dim = input_dim
#         self.batch_size = batch_size
#         self.output_dim=output_dim

#         # Feature Extraction - Small kernel to save timesteps
#         self.layer1 = nn.Conv1d(in_channels=self.input_dim, out_channels=50, kernel_size=20, stride=10)

#         # Dimensional Reduction Layer
#         self.layer2 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=599, stride=1)

#         # Output layers
#         self.preout = nn.Conv1d(in_channels=50, out_channels=self.output_dim, kernel_size=1)
#         self.out = nn.Sigmoid()

#         # Reusable layers
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()

#     def forward(self, x):

#         x = x.transpose(1,2).float()

#         x1 = self.layer1(x)
#         x1 = self.relu(x1)
#         x1 = self.dropout(x1)

#         x2 = self.layer2(x1)
#         x2 = self.relu(x2)
#         x2 = self.dropout(x2)

#         x3 = self.preout(x2)
#         x_out = self.out(x3)

#         return x_out.view(self.batch_size)

#     def model_type(self):
#         return "CNN"


""" CNN for 4 sec signals @ 128Hz = 512 samples"""
class CNN1D(nn.Module):

    def __init__(self, input_dim, batch_size, dropout, wantscuda, output_dim=1):
        super(CNN1D, self).__init__()

        self.input_dim = input_dim
        self.batch_size = batch_size
        self.output_dim=output_dim

        # Feature Extraction - Small kernel to save timesteps
        self.layer1 = nn.Conv1d(in_channels=self.input_dim, out_channels=15, kernel_size=5, stride=1)

        # Dimensional Reduction Layer
        self.layer2 = nn.Conv1d(in_channels=15, out_channels=15, kernel_size=508, stride=1)

        # Output layers
        self.preout = nn.Conv1d(in_channels=15, out_channels=self.output_dim, kernel_size=1)
        self.out = nn.Sigmoid()

        # Reusable layers
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.transpose(1,2).float()

        x1 = self.layer1(x)
        x1 = self.relu(x1)

        x2 = self.layer2(x1)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)

        x3 = self.preout(x2)
        x_out = self.out(x3)

        return x_out.view(self.batch_size)

    def model_type(self):
        return "CNN"
