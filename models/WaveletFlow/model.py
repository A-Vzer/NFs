import torch
import torch.nn as nn
import numpy as np
from modules.Wavelets.Haar.squeezeSplit import SqueezeSplit
from models.WaveletFlow.flowUnit import FlowUnit
from utilities.utils import uniform_binning_correction, split_feature
from modules.layers import gaussian_likelihood
import math


class WaveletFlow(nn.Module):
    def __init__(self, conditioning_network, params, level=-1):
        super().__init__()
        self.n_levels = params.nLevels
        self.base_level = params.baseLevel
        self.partial_level = level
        self.device = params.device
        self.haar_squeeze_split = SqueezeSplit(compensate=True, device=params.device)
        if self.partial_level == -1 or self.partial_level == self.base_level:
            params.K = params.stepsPerResolution[self.base_level]
            # self.spatial_biasing = params.spatialBiasing  # i dont know what this does (yet)
            self.base_flow = FlowUnit(params, params.imShape, self.base_level)
        else:
            self.base_flow = None

        start_flow_padding = nn.ModuleList([None] * self.base_level)  # add padding, since base may not be 0
        self.sub_flows = start_flow_padding.append(self.base_flow)  # append base
        for level in range(self.base_level + 1, self.n_levels + 1):
            params.K = params.stepsPerResolution[level]
            if self.partial_level != -1 and self.partial_level != level:
                self.sub_flows.append(None)
            else:
                H = 2 ** (level - 1)  # assume shape is always square
                W = 2 ** (level - 1)
                print(H, W)
                self.sub_flows.append(
                    FlowUnit(params, [9, H, W], level, conditional=True))

        self.conditioning_network = conditioning_network


    def forward(self, x, partial_level=-1, reverse=False):
        latents = []
        base = x
        log_density = 0
        if reverse:
            base_data = self.base_flow.latent_to_data(latents[self.base_level])
            base = base_data.data
            start_padding = [None] * self.base_level
            reconstructions = start_padding + [base]
            details = start_padding + [None]
            ld = base_data.ld
            ld_base = base_data.ld_base
            ldj = base_data.ldj
            for level in range(self.base_level + 1, self.n_levels + 1):
                latent = latents[level]
                base = reconstructions[-1]
                super_res = self.latent_to_super_res(latent, level, base)

                ld += super_res.ld
                ld_base += super_res.ld_base
                ldj += super_res.ldj
                reconstructions.append(super_res.reconstruction)
                details.append(super_res.details)

            return reconstructions
        else:
            for level in range(self.n_levels, self.base_level - 1, -1):
                # compute flow
                if level == partial_level or partial_level == -1:
                    if level == self.base_level:  # base level doesn't need to extract details
                        base, logdet = uniform_binning_correction(base)
                        flow = self.base_flow
                        latent, logdet = flow.forward(base, logdet)
                    else:
                        # decompose base
                        haar = self.haar_squeeze_split.forward(base)
                        details = haar.details
                        base = haar.base
                        details, logdet = uniform_binning_correction(details)
                        # condition
                        conditioning = self.conditioning_network.encoder_list[level](base)
                        flow = self.sub_flows[level]
                        latent, logdet = flow.forward(details, logdet, conditioning=conditioning)

                    latents.append(latent)
                    haar_logdet = torch.full(logdet.shape, flow.shape[0] * flow.shape[1] * flow.shape[2] * np.log(0.5) *
                                          (self.n_levels - level), device=self.device)
                    log_density += logdet + haar_logdet  # need custom haar_ldj because of partial
                    bpd = (-log_density) / (np.log(2.0) * flow.shape[0] * flow.shape[1] * flow.shape[2])
                    if partial_level != -1:
                        break  # stop of we are doing partial
                else:
                    # decompose base
                    if self.partial_level <= 8 and level > 8:  # i think this is where we use pre downsampled data
                        pass
                    else:  # perform dowsampling, but don't build flow
                        haar = self.haar_squeeze_split.forward(base)
                        base = haar.base

                    latents.append(None)

            return latents, bpd, None

    def sample_latents(self, n_batch=1, temperature=1.0):
        latents = [None] * self.base_level

        for flow in self.sub_flows:
            latents.append(flow.sample_latents(n_batch=n_batch, temperature=temperature))

        return latents


