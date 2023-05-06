import torch


class DycsNet(torch.nn.Module):
    """ modular net for dycs
    """

    def __init__(self, args, nets, master_net=None):
        super().__init__()
        self.nets = torch.nn.ModuleList(nets)
        self.num_classes = 1000
        self.args = args
        for i in range(len(nets)):
            self.nets[i] = self.nets[i].cuda()
        self.master_net = master_net

    def forward(self, x):
        if self.args.dycs_regime == 'concatenate':
            # forward pass through all networks
            ys = []
            for net in self.nets:
                y = net(x)
                ys.append(y)

            # concatenate all outputs
            # take only ags.dycs_classes_per_group
            # first elements of output vector for each networks
            if self.args.dycs_meaning_neurons == 'first':
                ys = [y[:, :self.args.dycs_classes_per_group] for y in ys]
            elif self.args.dycs_meaning_neurons == 'inplace':
                ys = [ys[i][:, i*self.args.dycs_classes_per_group:(
                    i+1)*self.args.dycs_classes_per_group] for i in range(len(ys))]
            y = torch.cat(ys, dim=1)

        elif self.args.dycs_regime == 'masternet':
            #use prediction of masternet to choose subnet
            ym = self.master_net(x)
            index = ym.argmax()
            net_index = (index / self.args.dycs_classes_per_group).int()
            y = self.nets[i](x)
        else:
            raise Exception('unknown dycs_regime')
        return y
