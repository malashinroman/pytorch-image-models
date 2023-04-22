import torch


class DycsNet(torch.nn.Module):
    """ modular net for dycs
    """
    def __init__(self, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.num_classes = 1000
        for i in range(len(nets)):
            self.nets[i] = self.nets[i].cuda()
            # self.nets[i] = self.nets[i].to('cuda:0')

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
        return y


