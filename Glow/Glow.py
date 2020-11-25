import math
import torch
import torch.nn as nn
from layers import SqueezeLayer, ActNorm2d, InvertibleConv1x1, Permute2d, Conv2d, Split2d, gaussian_sample, \
    gaussian_likelihood, gaussian_p
from tools import splitter, compute_same_pad, uniform_binning_correction

def get_block(in_channels, out_channels, hidden_channels):
    block = nn.Sequential(
        Conv2d(in_channels, hidden_channels),
        nn.ReLU(inplace=False),
        Conv2d(hidden_channels, hidden_channels, kernel_size=(1, 1)),
        nn.ReLU(inplace=False),
        Conv2dZeros(hidden_channels, out_channels),
    )
    return block


class FlowUnit(nn.Module):
    def __init__(self, inUnits, hiddenUnits, actNormScale, perm, coupling, LU):
        super().__init__()
        self.coupling = coupling
        # Actnorm
        self.actnorm = ActNorm2d(inUnits, actNormScale)

        # Permuatation
        if perm == "invconv":
            self.invconv = InvertibleConv1x1(inUnits, LU=LU)
            self.perm = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif perm == "shuffle":                                                                                                     self.shuffle = Permute2d(inUnits, shuffle=True)                                                                         self.perm = lambda z, logdet, rev: (self.shuffle(z, rev), logdet,)                                                  else:                                                                                                                       self.reverse = Permute2d(inUnits, shuffle=False)                                                                        self.perm = lambda z, logdet, rev: (self.reverse(z, rev), logdet,)                                                                                                                                                                          # Coupling                                                                                                              if coupling == "additive":                                                                                                  self.block = get_block(inUnits // 2, inUnits // 2, hiddenUnits)                                                     elif coupling == "affine":                                                                                                  self.block = get_block(inUnits // 2, inUnits, hiddenUnits)                                                                                                                                                                              def forward(self, inputt, logdet=None, reverse=False):
        if not reverse:
            return self.flow(inputt, logdet)
        else:
            return self.flow_r(inputt, logdet)

    # Forward pass
    def flow(self, inputt, logdet):
        # Actnorm
        z, logdet = self.actnorm(inputt, logdet, reverse=False)

        # Permute
        z, logdet = self.perm(z, logdet, False)

        # Coupling
        za, zb = splitter(z, 'split')
        if self.coupling == 'additive':
            zb = zb + self.block(za)
        elif self.coupling == 'affine':
            h = self.block(za)
            shift, scale = splitter(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            zb = zb + shift
            zb = zb * scale
            logdet = torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((za, zb), dim=1)
        return z, logdet

    # Reverse pass
    def flow_r(self, inputt, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z1, z2 = splitter(inputt, "split")
        if self.flow_coupling == "additive":
            z2 = z2 - self.block(z1)
        elif self.flow_coupling == "affine":
            h = self.block(z1)
            shift, scale = splitter(h, "cross")
            scale = torch.sigmoid(scale + 2.0)
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=[1, 2, 3]) + logdet
        z = torch.cat((z1, z2), dim=1)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet, reverse=True)

        return z, logdet


class MultiScaleFlow(nn.Module):
    def __init__(self, imageDim, hiddenUnits, K, L, actNormScale, perm, coupling, LU):
        super().__init__()
        self.layers = nn.ModuleList()  # properly registered list of modules
        self.outputDim = []

        self.K = K
        self.L = L
        H, W, C = imageDim

        # Build squeeze --> K x flow --> split loop L times
        for i in range(L):
            # Squeeze
            C, H, W = C * 4, H // 2, W // 2  # new dimensions
            self.layers.append(SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # Flow
            for _ in range(K):
                self.layers.append(FlowUnit(inUnits=C, hiddenUnits=hiddenUnits, actNormScale=actNormScale,
                                            perm=perm, coupling=coupling, LU=LU))

            self.output_shapes.append([-1, C, H, W])

            # Split
            if i < L - 1:
                self.layers.append(Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, z, logdet=0.0, reverse=False):
        if reverse:
            for layer in reversed(self.layers):
                if isinstance(layer, Split2d):
                    z, logdet = layer(z, logdet=0, reverse=True)
                else:
                    z, logdet = layer(z, logdet=0, reverse=True)
            return z
        else:
            for layer, shape in zip(self.layers, self.output_shapes):
                z, logdet = layer(z, logdet, reverse=False)
            return z, logdet


class Glow(nn.Module):
    # imageDim: Dimensions of input images
    # hiddenUnits: Number of units in a single hidden layer
    # K: Depth (number of actNorm -> 1x1 conv -> affine coupling) of a single flow
    # L: Levels, or number of times (L-1) the sqeeze-step-split process occurs before final sqeeze-step
    # actNormScale: Scale parameter of actnorm layer
    # perm: 1x1 convolution
    # coupling: Affine coupling layer
    # LU: LU decomposition of weight matrix W
    def __init__(self, imageDim, hiddenUnits, K, L, actNormScale, perm, coupling, LU):
        super().__init__()
        self.flow = MultiScaleFlow(imageDim, hiddenUnits, K, L, actNormScale, perm, coupling, LU)

        self.register_buffer("prior_h", torch.zeros([1, self.flow.output_shapes[-1][1] * 2,
                                                     self.flow.output_shapes[-1][2],
                                                     self.flow.output_shapes[-1][3]]), )

    def prior(self, data):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(32, 1, 1, 1)

        return splitter(h, "split")

    def forward(self, x=None, z=None, reverse=False):
        if reverse:
            return self.reverse_flow(z)
        else:
            return self.normal_flow(x)

    def normal_flow(self, x):
        b, c, h, w = x.shape

        x, logdet = uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)

        mean, logs = self.prior(x)
        objective += gaussian_likelihood(mean, logs, z)

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd

    def reverse_flow(self, z):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z)
                z = gaussian_sample(mean, logs)
            x = self.flow(z, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, ActNorm2d):
                m.inited = True