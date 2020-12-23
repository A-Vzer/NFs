from Glow import model, glow, tools
from WaveletGlow import waveletGlow
import torch.optim as optim
import torch



class Adapter:
    def __init__(self, modelName, imShape, device):
        if modelName == 'glow':
            self.flow = glow.Parameters(imShape, device)
            self.modelName = modelName
        if modelName == 'waveletglow':
            self.flow = waveletGlow.Parameters(imShape, device)
            self.modelName = modelName

