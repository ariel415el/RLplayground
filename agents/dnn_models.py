import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        super(MLP, self).__init__()
        layers = [torch.nn.Linear(input_dim, hidden_layer_sizes[0]), torch.nn.ReLU()]

        for i in range(1, len(hidden_layer_sizes)):
            layers += [torch.nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]), torch.nn.ReLU()]

        layers += [torch.nn.Linear(hidden_layer_sizes[-1], output_dim)]

        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = x.double()
        x = self.model(x)
        return x

class MLP_softmax(MLP):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        MLP.__init__(self, input_dim, output_dim, hidden_layer_sizes)

    def forward(self, x):
        x = super().forward(x)
        x = torch.nn.functional.softmax(x)
        return x
#
# class combined_MLP(torch.nn.Module):
#     def __init__(self, input_dim, output_dim,, ha_1, ha_2, hb_1, hb_2):
#         self.softmax_mlp = MLP_softmax(input_dim, output_dim, ha_1, ha_2)
#         self.mlp = MLP_softmax(input_dim, output_dim, hb_1, hb_2)
#
#     def forward(self, x):
#         return self.softmax_mlp(x), self.mlp(x)