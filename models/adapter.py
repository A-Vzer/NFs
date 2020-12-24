from models.Glow.parameters import GlowParameters
from models.Glow.scripts import RunGlow
from models.WaveletFlow.parameters import WaveletFlowParameters
from models.WaveletFlow.scripts import RunWaveletFlow


class Adapter:
    def __init__(self, modelName, imShape, device, level=None):
        if modelName == 'glow':
            self.flow = GlowParameters(imShape, device)
            self.scripts = RunGlow(self.flow.model, device)
        if modelName == 'waveletglow':
            self.flow = WaveletFlowParameters(level, imShape, device)
            self.scripts = RunWaveletFlow(self.flow.model, device)

