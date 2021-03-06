#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from torch import nn
# from nni.retiarii.nn.pytorch import nn

class RNN(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, sequence_length):
        super(RNN, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        self.rnn = nn.LSTM(input_size,hidden_size,num_layers, batch_first=True)
        # self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size * self.sequence_length, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.output_size)
        )
    def forward(self, inputX):
        rnn_output, _ = self.rnn(inputX)
        rnn_output = rnn_output.contiguous().view(-1, self.hidden_size * self.sequence_length)
        output = self.mlp(rnn_output)
        output = output.contiguous().view(-1, self.output_size)
        return output
