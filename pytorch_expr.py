import copy
import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(2, 16)
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(16, 16)
        self.bn = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, 16)

    def forward(self, x):
        x = self.input(x)
        x = self.act(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.output(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = BaseModel()
        self.classifier = nn.Linear(16, 1)

    def forward(self, x):
        x = self.feature(x)
        return self.classifier(x)

def test(freeze_feature: bool, not_optimize_feature: bool, eval_mode_feature: bool):
    test_input_1 = torch.rand((128, 2))
    test_input_2 = torch.rand((128, 2))

    model = Model()
    initial_model = copy.deepcopy(model)

    model.eval()
    initial_feature_1 = model.feature(test_input_1)

    if freeze_feature:
        model.feature.requires_grad_(False)

    params_to_optimizer = model.classifier.parameters() if not_optimize_feature else model.parameters()
    optimizer = torch.optim.SGD(params_to_optimizer, lr=0.001)

    model.train()
    if eval_mode_feature:
        model.feature.eval()
    optimizer.zero_grad()
    y = model(test_input_2)
    loss = y.sum()
    loss.backward()
    optimizer.step()

    model.eval()
    feature_1 = model.feature(test_input_1)

    params_changed = any(
        (p1 != p2).any()
        for p1, p2 in zip(model.feature.parameters(), initial_model.feature.parameters())
    )
    bn_changed = ((initial_model.feature.bn.running_mean != model.feature.bn.running_mean).any() or \
        (initial_model.feature.bn.running_var != model.feature.bn.running_var).any()).item()

    print(params_changed, bn_changed)


if __name__ == "__main__":
    test(False, False, True)
    test(True, False, True)
    test(False, True, True)
    test(True, True, True)
    test(True, True, False)
    test(False, False, False)
