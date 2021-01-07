import torch
import torch.optim as optim
from parameters import Parameters
from WaveletFlow.networkBody import Network
from utilities.utils import loss


class WaveletFlowParameters(Parameters):
    def __init__(self, level, imShape, device):
        super().__init__(imShape, device)
        self.modelName = 'waveletflow'
        self.spatialBiasing = [False, False, False, False, False, False, False]  # not implemented yet
        self.stepsPerResolution = [8, 8, 16, 16, 16, 16, 16]
        self.cropFactor = [1, 1, 1, 1, 1, 2, 2]
        self.n_batch = [64, 64, 64, 64, 64, 64, 64]
        self.n_ddi_batch = [64, 64, 64, 64, 64, 64, 64]
        self.nLevels = 6
        self.kernel = 3
        self.baseLevel = 0
        self.partialLevel = -1
        self.convWidth = [128, 128, 128, 128, 128, 256, 256]
        self.hiddenChannels = 256
        self.n_res_blocks = 3
        self.lr = 1e-2
        self.weight_decay = 1
        self.lr_lambda = lambda epoch: min(1.0, (epoch + 1) / self.warmup)  # noqa
        self.zero_use_logscale = True
        self.normalize = True
        self.actNormScale = 3.0
        self.zero_logscale_factor = 3.0
        self.y_condition = False  # not implemented
        self.model = Network(level, self).to(device)
        self.optimizer = optim.Adamax(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # for param in self.model.parameters():
        #     param.requires_grad = True



