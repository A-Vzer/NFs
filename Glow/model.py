import math
import torch
import torch.nn as nn
from Glow import layers as l
from Glow import tools as t
from ResNet import resnet
import torch.nn.functional as F
from Coupling.basicCoupling import Additive, Affine, SoftCheckerboard
from Coupling.cycleMask import CycleMask


class FlowStep(nn.Module):
    def __init__(self, in_channels, hidden_channels, actnorm_scale, flow_permutation, flow_coupling, LU_decomposed, device):
        super().__init__()
        self.flow_coupling = flow_coupling
        self.device = device
        self.actnorm = l.ActNorm2d(in_channels, actnorm_scale)
        self.rescale = nn.utils.weight_norm(Rescale(in_channels))
        self.eps = 1e-6
        
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = l.InvertibleConv1x1(in_channels, LU_decomposed=LU_decomposed)
            self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)
        elif flow_permutation == "shuffle":
            self.shuffle = l.Permute2d(in_channels, shuffle=True)
            self.flow_permutation = lambda z, logdet, rev: (self.shuffle(z, rev), logdet,)
        else:
            self.reverse = l.Permute2d(in_channels, shuffle=False)
            self.flow_permutation = lambda z, logdet, rev: (self.reverse(z, rev), logdet,)

        # 3. coupling
        if flow_coupling == "additive":
            self.coupling = Additive(in_channels // 2, in_channels // 2, hidden_channels)
        elif flow_coupling == "affine":
            self.coupling = Additive(in_channels // 2, in_channels, hidden_channels)
        elif flow_coupling == "checker":
            self.coupling = SoftCheckerboard(in_channels, in_channels * 2, hidden_channels, self.device)
        elif flow_coupling == "cycle":
            self.coupling = CycleMask(in_channels, in_channels * 2, hidden_channels, self.device)

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, logdet)
        else:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1. actnorm
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, False)

        # 3. coupling
        z, logdet = self.coupling(z, logdet, False)

        return z, logdet

    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        # 1.coupling
        z, logdet = self.coupling(input, logdet, True)

        # 2. permute
        z, logdet = self.flow_permutation(z, logdet, True)

        # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet


class FlowNet(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L, actnorm_scale, flow_permutation, flow_coupling,
            LU_decomposed, device):
        super().__init__()

        self.layers = nn.ModuleList()
        self.output_shapes = []

        self.K = K
        self.L = L

        H, W, C = image_shape

        for i in range(L):
            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(l.SqueezeLayer(factor=2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for _ in range(K):
                self.layers.append(FlowStep(in_channels=C, hidden_channels=hidden_channels, actnorm_scale=actnorm_scale,
                    flow_permutation=flow_permutation, flow_coupling=flow_coupling, LU_decomposed=LU_decomposed, device=device))
                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if i < L - 1:
                self.layers.append(l.Split2d(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2

    def forward(self, input, logdet=0.0, reverse=False, temperature=None):
        if reverse:
            return self.decode(input, temperature)
        else:
            return self.encode(input, logdet)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, temperature=None):
        for layer in reversed(self.layers):
            if isinstance(layer, l.Split2d):
                z, logdet = layer(z, logdet=0, reverse=True, temperature=temperature)
            else:
                z, logdet = layer(z, logdet=0, reverse=True)
        return z


class Glow(nn.Module):
    def __init__(self, image_shape, hidden_channels, K, L, actnorm_scale, flow_permutation, flow_coupling,
            LU_decomposed, y_classes, learn_top, y_condition, device):
        super().__init__()
        self.flow = FlowNet(image_shape=image_shape, hidden_channels=hidden_channels, K=K, L=L,
            actnorm_scale=actnorm_scale, flow_permutation=flow_permutation, flow_coupling=flow_coupling,
            LU_decomposed=LU_decomposed, device=device)
        self.y_classes = y_classes
        self.y_condition = y_condition

        self.learn_top = learn_top

        # learned prior
        if learn_top:
            C = self.flow.output_shapes[-1][1]
            self.learn_top_fn = l.Conv2dZeros(C * 2, C * 2)

        if y_condition:
            C = self.flow.output_shapes[-1][1]
            self.project_ycond = l.LinearZeros(y_classes, 2 * C)
            self.project_class = l.LinearZeros(C, y_classes)

        self.register_buffer("prior_h", torch.zeros(
            [1, self.flow.output_shapes[-1][1] * 2, self.flow.output_shapes[-1][2],
                self.flow.output_shapes[-1][3], ]), )

    def prior(self, data, y_onehot=None):
        if data is not None:
            h = self.prior_h.repeat(data.shape[0], 1, 1, 1)
        else:
            # Hardcoded a batch size of 32 here
            h = self.prior_h.repeat(32, 1, 1, 1)

        channels = h.size(1)

        if self.learn_top:
            h = self.learn_top_fn(h)

        if self.y_condition:
            assert y_onehot is not None
            yp = self.project_ycond(y_onehot)
            h += yp.view(h.shape[0], channels, 1, 1)
        return t.split_feature(h, "split")

    def forward(self, x=None, y_onehot=None, z=None, temperature=None, reverse=False):
        if reverse:
            return self.reverse_flow(z, y_onehot, temperature)
        else:
            return self.normal_flow(x, y_onehot)

    def normal_flow(self, x, y_onehot):
        b, c, h, w = x.shape

        x, logdet = t.uniform_binning_correction(x)

        z, objective = self.flow(x, logdet=logdet, reverse=False)
        mean, logs = self.prior(x, y_onehot)
        objective += l.gaussian_likelihood(mean, logs, z)
        if self.y_condition:
            y_logits = self.project_class(z.mean(2).mean(2))
        else:
            y_logits = None

        # Full objective - converted to bits per dimension
        bpd = (-objective) / (math.log(2.0) * c * h * w)

        return z, bpd, y_logits

    def reverse_flow(self, z, y_onehot, temperature):
        with torch.no_grad():
            if z is None:
                mean, logs = self.prior(z, y_onehot)
                z = l.gaussian_sample(mean, logs, temperature)
            x = self.flow(z, temperature=temperature, reverse=True)
        return x

    def set_actnorm_init(self):
        for name, m in self.named_modules():
            if isinstance(m, l.ActNorm2d):
                m.inited = True


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_channels):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_channels, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x