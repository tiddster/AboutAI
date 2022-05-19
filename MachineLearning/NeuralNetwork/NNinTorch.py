import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize as opt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(20 * 20, 25)
        self.fc2 = nn.Linear(25, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)

costFunc = nn.NLLLoss()

net = Net()
print(net)
