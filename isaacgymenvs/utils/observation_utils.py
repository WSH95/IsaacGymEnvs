from isaacgymenvs.utils.circle_buffer import CircleBuffer
import torch
from typing import Union


class ObservationBuffer:
    def __init__(self,
                 num_envs: int,
                 single_data_shape: tuple,
                 data_type: torch.dtype,
                 buffer_length: int,
                 device: str = "cuda:0",
                 scale: Union[float, list] = 1.0,
                 noise: Union[float, list, None] = None):
        self.obs_raw_buffer = CircleBuffer(num_envs, single_data_shape, data_type, buffer_length, device)
        self.obs_noisy_scaled_buffer = CircleBuffer(num_envs, single_data_shape, data_type, buffer_length, device)
        self.scale = scale
        if isinstance(scale, list):
            self.scale = torch.tensor(scale, dtype=torch.float, device=device, requires_grad=False)
        self.noise = noise  # Standard deviation
        if isinstance(noise, list):
            self.noise = torch.tensor(noise, dtype=torch.float, device=device, requires_grad=False)
        elif noise == 0.:
            self.noise = None

    def reset_and_fill(self, data: torch.Tensor):
        self.obs_raw_buffer.reset_and_fill(data)
        if self.noise is not None:
            self.obs_noisy_scaled_buffer.reset_and_fill((data + torch.randn_like(data) * self.noise) * self.scale)
        else:
            self.obs_noisy_scaled_buffer.reset_and_fill(data * self.scale)

    def reset_and_fill_index(self, idx: Union[list, torch.Tensor], data: torch.Tensor):
        self.obs_raw_buffer.reset_and_fill_index(idx, data)
        if self.noise is not None:
            self.obs_noisy_scaled_buffer.reset_and_fill_index(idx, (data + torch.randn_like(data) * self.noise) * self.scale)
        else:
            self.obs_noisy_scaled_buffer.reset_and_fill_index(idx, data * self.scale)

    def record(self, data: torch.Tensor):
        self.obs_raw_buffer.record(data)
        if self.noise is not None:
            self.obs_noisy_scaled_buffer.record((data + torch.randn_like(data) * self.noise) * self.scale)
        else:
            self.obs_noisy_scaled_buffer.record(data * self.scale)

    def get_latest_data(self):
        return self.obs_noisy_scaled_buffer.get_latest_data()

    def get_latest_data_raw(self):
        return self.obs_raw_buffer.get_latest_data()

    def get_index_data(self, indices_from_back: Union[torch.Tensor, list, int]):
        return self.obs_noisy_scaled_buffer.get_index_data(indices_from_back)

    def get_index_data_raw(self, indices_from_back: Union[torch.Tensor, list, int]):
        return self.obs_raw_buffer.get_index_data(indices_from_back)
