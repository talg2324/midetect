import torch
import torch.nn as nn

class LSTM(nn.Module):
 
    def __init__(self, input_dim, hidden_dim, batch_size, dropout, wantscuda, num_layers=1, output_dim=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.output_dim=output_dim
        
        # Whether or not to use GPU
        self.wantscuda = wantscuda
 
        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, bidirectional=True, dropout=dropout)

        # Define the model's computations
        self.linear = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.dropout = nn.Dropout(dropout)

        # Define the output layer
        self.out = nn.Sigmoid()
 
    def init_hidden(self):
        device = torch.device('cuda:0' if self.wantscuda else 'cpu')
        # This is what we'll initialise our hidden state as
        return (torch.randn(self.num_layers*2, self.batch_size, self.hidden_dim).to(device),
                torch.randn(self.num_layers*2, self.batch_size, self.hidden_dim).to(device))
 
    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).

        
        input = input.transpose(1, 0).float()
        
        h0 = self.init_hidden()

        lstm_out, (hn, cn) = self.lstm(input.view(-1, self.batch_size, self.input_dim), h0)

        lstm_out = torch.cat([hn[-1,:,:], hn[-2,:,:]], 1)

        lstm_out = self.dropout(lstm_out)

        #  hidden dims x batch size -> batch size x 1 (squeeze hidden dims)
        y_pred = self.linear(lstm_out)

        y_pred = self.out(y_pred)

        return y_pred.squeeze()

    def model_type(self):
        return "RNN"
 
