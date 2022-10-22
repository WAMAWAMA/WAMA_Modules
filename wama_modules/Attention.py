





class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=None, which_first=0, dim=2):
        super().__init__()
        self.which_first = which_first
        self.dim = dim

        if reduction is None:
            reduction = in_channels

        if in_channels % reduction != 0:
            raise ValueError('in_channels % reduction should be 0')

        if self.dim == 2:
            make_conv = nn.Conv2d
            pooling = nn.AdaptiveAvgPool2d
        elif self.dim == 3:
            make_conv = nn.Conv3d
            pooling = nn.AdaptiveAvgPool3d

        self.cSE = nn.Sequential(
            pooling(1),
            make_conv(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            make_conv(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(make_conv(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        if self.which_first == 0:
            return x * self.cSE(x) + x * self.sSE(x)
        elif self.which_first == 1:
            x = x * self.cSE(x)
            return x * self.sSE(x)
        elif self.which_first == 2:
            x = x * self.sSE(x)
            return x * self.cSE(x)





# nonlocal