import torch
from typing import Union, List
from omegaconf.listconfig import ListConfig


class CircleBuffer:
    def __init__(self, num_buckets: int, single_data_shape: tuple, data_type: torch.dtype, buffer_length: int, device: str = "cuda:0"):
        self.buffer_shape = (buffer_length, num_buckets,) + single_data_shape
        self.single_data_shape = single_data_shape
        self.data_shape = (num_buckets,) + single_data_shape
        self.data_dtype = data_type
        self.device = device
        self.buffer = torch.zeros(self.buffer_shape, dtype=data_type, device=device, requires_grad=False)
        self.buffer_len: int = buffer_length
        self.front: int = 0
        self.rear: int = self.buffer_len - 1

        self.permute_list = list(range(len(self.buffer_shape)))
        self.permute_list[0] = 1
        self.permute_list[1] = 0

    def reset_and_fill(self, data: torch.Tensor):
        assert data.shape == self.data_shape
        self.front = 0
        self.rear = self.buffer_len - 1
        # for i in range(self.buffer_len):
        #     self.buffer[i] = data.clone().to(self.device)
        self.buffer[:] = data.clone().to(self.device).to(self.data_dtype).unsqueeze(dim=0).repeat_interleave(self.buffer_len, 0)

    def reset_and_fill_index(self, idx: Union[List, ListConfig, torch.Tensor], data: torch.Tensor):
        assert isinstance(idx, (List, ListConfig, torch.Tensor))
        assert (len(idx),) + self.single_data_shape == data.shape
        if isinstance(idx, (List, ListConfig)):
            idx = torch.tensor(idx, dtype=torch.long)
        else:
            idx = idx.to(torch.long)
        self.buffer[:, idx, :] = data.clone().to(self.device).to(self.data_dtype).unsqueeze(dim=0).repeat_interleave(self.buffer_len, 0)

    def record(self, data: torch.Tensor):
        assert data.shape == self.data_shape
        self.front = (self.front + 1) % self.buffer_len
        self.rear = (self.rear + 1) % self.buffer_len

        self.buffer[self.rear] = data.clone().to(self.device).to(self.data_dtype)

    def get_latest_data(self):
        return self.buffer[self.rear].clone()

    def get_index_data(self, indices_from_back):
        assert isinstance(indices_from_back, (torch.Tensor, List, ListConfig, int))
        if isinstance(indices_from_back, torch.Tensor):
            indices = ((self.rear - indices_from_back + self.buffer_len) % self.buffer_len).to(torch.long)
        elif isinstance(indices_from_back, (List, ListConfig)):
            indices = ((self.rear - torch.tensor(indices_from_back, dtype=torch.long) + self.buffer_len) % self.buffer_len).to(torch.long)
        else:
            indices = ((self.rear - torch.tensor(indices_from_back, dtype=torch.long).unsqueeze(0) + self.buffer_len) % self.buffer_len).to(torch.long)

        return self.buffer[indices].clone().permute(*self.permute_list).flatten(1, 2)

    def get_len_data(self, length: int):
        assert 0 < length < self.buffer_len + 1
        if self.rear == self.buffer_len - 1:
            return self.buffer[-length:].clone().permute(1, 2, 0)
        else:
            return torch.cat((self.buffer[self.rear + 1:], self.buffer[:self.rear + 1]), dim=0)[-length:].permute(1, 2, 0)


if __name__ == "__main__":
    buffer = CircleBuffer(3, (2,), torch.float32, 5)
    init = torch.Tensor([[1., 1.], [2., 2.], [3., 3.]])
    buffer.reset_and_fill(init)
    print(buffer.buffer)
    buffer.reset_and_fill_index(torch.Tensor([0, 2]), torch.Tensor([[0., 0.], [0., 0.]]))
    print(buffer.buffer)
    a = torch.Tensor([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]])
    b = torch.Tensor([[5.5, 5.5], [6.6, 6.6], [7.7, 7.7]])
    print(buffer.get_latest_data())
    buffer.record(a)
    print(buffer.get_latest_data())
    buffer.record(b)
    print(buffer.get_latest_data())
    buffer.record(a + 1)
    print(buffer.get_latest_data())
    buffer.record(b + 1)
    print(buffer.get_latest_data())
    buffer.record(a + 2)
    print(buffer.get_latest_data())
    buffer.record(b + 2)
    print(buffer.get_latest_data())
    print(buffer.get_index_data(2))
    print(buffer.get_index_data([0, 1, 3]))
    print(buffer.get_index_data(torch.Tensor([2, 1, 0])))
    print(buffer.get_len_data(1))
