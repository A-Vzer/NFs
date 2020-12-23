from models.Glow.parameters import GlowParameters
from models.Glow.scripts import RunGlow
from models.WaveletFlow.parameters import WaveletFlowParamters

class Adapter:
    def __init__(self, modelName, imShape, device):
        if modelName == 'glow':
            self.flow = GlowParameters(imShape, device)
            self.scripts = RunGlow(self.flow.model, device)
        if modelName == 'waveletglow':
            self.flow = WaveletFlowParamters(imShape, device)

