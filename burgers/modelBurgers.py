import torch


class ModelBurgers(torch.nn.Module):
    def __init__(self):
        super(ModelBurgers, self).__init__()

        self.layers = []
        self.layers.append(torch.nn.Linear(2, 32))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(32, 16))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(16, 32))
        self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(32, 1))

        self.net = torch.nn.Sequential(*self.layers)

    def forward(self, input_chunk):
        output = self.net(input_chunk)
        return output
