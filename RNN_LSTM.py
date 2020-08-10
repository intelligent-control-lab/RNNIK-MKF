import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):
    def __init__(self, input_dim, input_step, hidden_layer_size, num_layers, output_step, output_dim, device):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=hidden_layer_size, 
                            num_layers=num_layers, 
                            batch_first=True)

        self.hidden_size = hidden_layer_size
        self.num_layers = num_layers
        self.linear = nn.Linear(self.hidden_size, output_step*output_dim)
        self.sequence_length = input_step
        self.output_step = output_step
        self.output_dim = output_dim
        self.device = device

        # Params for adaptation
        self.x_pre = 0
 
    def forward(self, input_seq):
        self.hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device),
                            torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device))
        
        lstm_out, _ = self.lstm(input_seq, self.hidden_cell)
        xt = lstm_out[:, -1, :].view(self.hidden_size, 1).cpu()

        # Take the output from last RNN layer
        self.x_pre = xt
        
        # Take the last output
        predictions = self.linear(lstm_out).view(self.sequence_length, self.output_step*self.output_dim)[-1]
        predictions = predictions.view(self.output_step, self.output_dim)
        return predictions