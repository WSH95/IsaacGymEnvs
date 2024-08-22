# -*- coding: utf-8 -*-
# Created by Shuhan Wang on 2024/6/6.
#

import torch
import torch.nn as nn
from rl_games.algos_torch import network_builder


class CustomBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)
            return

        def load(self, params):
            super().load(params)
            return

        def forward(self, input_dict):
            return

    def build(self, name, **kwargs):
        net = CustomBuilder.Network(self.params, **kwargs)
        return net
