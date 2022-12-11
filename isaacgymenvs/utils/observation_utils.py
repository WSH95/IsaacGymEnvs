from isaacgymenvs.utils.circle_buffer import CircleBuffer
import torch


class ObservationBuffer:
    def __init__(self,
                 data_shape: tuple,
                 data_type: torch.dtype,
                 buffer_length: int,
                 device: str = "cuda:0",
                 noise=None):
        self.obs_raw_buffer = CircleBuffer(data_shape, data_type, buffer_length, device)
        self.obs_noisy_buffer = CircleBuffer(data_shape, data_type, buffer_length, device)
        self.noise = noise
