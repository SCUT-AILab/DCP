import torch.nn as nn


class AuxClassifier(nn.Module):
    """
    define auxiliary classifier:
    BN->RELU->AVGPOOLING->FC
    """

    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels, num_classes)

        # init params
        self.fc.bias.data.zero_()

    def forward(self, x):
        """
        forward propagation
        """
        out = self.bn(x)
        out = self.relu(out)
        out = out.mean(2).mean(2)
        out = self.fc(out)
        return out
