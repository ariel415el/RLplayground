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
        # x = x.float()
        x = self.model(x)
        return x

class MLP_softmax(MLP):
    def __init__(self, input_dim, output_dim, hidden_layer_sizes):
        MLP.__init__(self, input_dim, output_dim, hidden_layer_sizes)

    def forward(self, x):
        x = super().forward(x)
        x = torch.nn.functional.softmax(x)
        return x


class ActorCritic(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hiddent_layer):
        super(ActorCritic, self).__init__()
        self.affine = torch.nn.Linear(input_dim, hiddent_layer)

        self.action_layer = torch.nn.Linear(hiddent_layer, output_dim)
        self.value_layer = torch.nn.Linear(hiddent_layer, 1)

    def forward(self, x):
        features = torch.nn.functional.relu(self.affine(x))

        probs = torch.nn.functional.softmax(self.action_layer(features))
        value = self.value_layer(features)

        return probs, value