class ResidualBlock(nn.Module):
    """
    实现子module: Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.se = SELayer(outchannel, 16)
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        out = self.se(out)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)