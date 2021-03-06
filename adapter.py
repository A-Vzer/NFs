from models.Glow.parameters import GlowParameters
from models.Glow import scripts as glow
from models.WaveletFlow.parameters import WaveletFlowParameters
from models.WaveletFlow import scripts as waveletglow


class Adapter:
    def __init__(self, modelName, imShape, device, level=None):
        if modelName == 'glow':
            self.flow = GlowParameters(imShape, device)
            self.scripts = glow.RunGlow(self.flow, device)
        if modelName == 'waveletglow':
            self.flow = WaveletFlowParameters(level, imShape, device)
            self.scripts = waveletglow.RunWaveletFlow(self.flow, device)

