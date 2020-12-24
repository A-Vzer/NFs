from models.WaveletFlow.model import WaveletFlow
from models.WaveletFlow.conditionNet import ConditioningNetwork


class Network(WaveletFlow):
    def __init__(self, level, params):
            conditioning_net = ConditioningNetwork()
            super().__init__(conditioning_net, params, level)