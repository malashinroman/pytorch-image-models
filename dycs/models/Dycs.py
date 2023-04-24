import torch


class DycsNet(torch.nn.Module):
    """ modular net for dycs
    """

    def __init__(self, args, nets):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.num_classes = 1000
        self.args = args
        for i in range(len(nets)):
            self.nets[i] = self.nets[i].cuda()

    def forward(self, x):
        # forward pass through all networks
        ys = []
        # x = x.to('cuda:0')
        for net in self.nets:
            y = net(x)
            ys.append(y)

        # concatenate all outputs
        # take only ags.dycs_classes_per_group
        # first elements of output vector for each networks
        __import__('pudb').set_trace()
        if self.args.dycs_meaning_neurons == 'first':
            ys = [y[:, :self.args.dycs_classes_per_group] for y in ys]
        elif self.args.dycs_meaning_neurons == 'inplace':
            ys = [y[i][:, i*self.args.dycs_classes_per_group:(
                i+1)*self.args.dycs_classes_per_group] for i in range(len(ys))]
        y = torch.cat(ys, dim=1)
        return y
