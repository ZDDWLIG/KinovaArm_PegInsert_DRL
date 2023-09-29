import torch.nn as nn
import torch


class LSTMModel(nn.Module):
    def __init__(self, input_size, feature_dim, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.trunk = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LeakyReLU(inplace=True))
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=feature_dim, num_layers=1, batch_first=True)
        self.norm = nn.Sequential(nn.LayerNorm(feature_dim), nn.Tanh())
        self.linear = nn.Sequential(nn.Linear(feature_dim, hidden_size),
                                    nn.LeakyReLU(inplace=True), nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x_ = self.trunk(x)
        _, (out, _) = self.lstm(x_)
        out = out[-1:].permute(1, 0, 2).squeeze(dim=1)
        out = self.norm(out) # feature
        out = self.linear(out)
        out = torch.tanh(out) # norm (-1, 1)
        return out

