from Glow import model, Glow, tools
import torch.optim as optim
import torch



class Adapter:
    def __init__(self, modelName, imShape, device):
        if modelName == 'glow':
            self.flow = Glow.Parameters(imShape, device)
            self.modelName = modelName
