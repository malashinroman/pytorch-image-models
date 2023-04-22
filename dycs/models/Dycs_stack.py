import torch


class DycsNetStack(torch.nn.Module):
    """ modular net for dycs
    """
    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.num_classes = 1000
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.RELU(),
            torch.nn.Linear(1000, 1000))
        for i in range(len(nets)):
            self.nets[i] = self.nets[i].cuda()

            # freeze all parameters, except classifier
            for param in self.nets[i].parameters():
                param.requires_grad = False

    def forward(self, x):
        # forward pass through all networks
        ys = []
        # x = x.to('cuda:0')
        for net in self.nets:
            y = net(x)
            ys.append(y)

        # concatenate all outputs
        # take only 100 first elements of output vector for each networks
        ys = [y[:, :100] for y in ys]
        y = torch.cat(ys, dim=1)
        y = self.classifier(y)

        return y


