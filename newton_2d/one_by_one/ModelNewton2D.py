import torch


class ModelNewton2D(torch.nn.Module):
    def __init__(self):
        super(ModelNewton2D, self).__init__()
        self.layers = []
        self.layers.append(torch.nn.Linear(1, 32))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(32, 16))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(16, 32))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(32, 5))
        self.net = torch.nn.Sequential(*self.layers)

    def forward(self, input_chunk):
        output = self.net(input_chunk)
        return output
