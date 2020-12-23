import torch
import torch.nn as nn
from WaveletGlow.multiStep import MultiStep
from Glow.layers import ActNorm2d
import numpy as np
import math
from Glow.tools import uniform_binning_correction, compute_loss


class FlowUnit(nn.Module):
    def __init__(self, params, shape, level, conditional=False):
        super().__init__()
        self.C = shape[0]
        self.H = shape[1]
        self.W = shape[2]
        self.shape = shape
        data_shape = [1, self.C, self.H, self.W]
        params.n_squeezes = 0
        self.device = params.device
        # generate flow layers here
        self.step = MultiStep(params, data_shape, level, conditional)
        # base measure tforms
        # if self.spatial_bias:
        #     self.base_tform = bijector.Element_shift_scale()
        # else:
        #     self.base_tform = bijector.Act_norm()
        self.base_tform = ActNorm2d(self.C).to(self.device)

    def forward(self, x, conditioning=None, reverse=False):
        z = x
        z, logdet = uniform_binning_correction(z)
        if reverse:
            latent, ldj_base = self.base_tform.forward(z, reverse)
            z, logdet = self.step.forward(x, conditioning, reverse)
            ld_base = self.latent_log_density(z)
            # ldj = - ldj_base - logdet
            # ld = ld_base + ldj
            return z
        else:
            z, logdet = self.step.forward(x, logdet, conditioning)
            z, logdet = self.base_tform.forward(z, logdet)
            ld = self.latent_log_density(z)
            log_density = logdet + ld
            bpd = (-log_density) / (math.log(2.0) * self.C * self.H * self.W)
            return z, bpd

    def latent_log_density(self, latent):
        '''
        latent variable is diagonal unit gaussian
        '''
        element_log_density = -0.5 * np.log(2 * np.pi)
        element_log_density -= 0.5 * latent ** 2
        total_log_density = torch.sum(element_log_density, axis=[1, 2, 3])

        return total_log_density

    def sample_latents(self, n_batch=1, temperature=1.0, truncate=False):
        '''
        samples all latents required to generate data(s)
        '''
        shapes = self.compute_latent_shapes(self.data_shape, self.n_squeezes)
        latents = []

        if not type(temperature) == list:
            temperature = [temperature] * len(shapes)

        for shape, temp in zip(shapes, temperature):
            latents.append(self.sample_latent([n_batch] + shape, temp, truncate))

        return latents

    def sample_latent(self, shape, temperature=1.0):
        latent = torch.random.normal(size=shape) * temperature
        return latent