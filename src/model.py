#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
# from torch import nn
# from nni.retiarii.nn.pytorch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self