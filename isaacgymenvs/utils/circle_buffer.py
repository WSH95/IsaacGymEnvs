import torch


class CircleBuffer:
    def __init__(self, data_shape: tuple, data_type: torch.dtype, buffer_length: int, device: str = "cuda:0"):
        buffer_shape = (buffer_length,) + data_shape
        self.data_shape = data_shape
        self.device = device
        self.buffer = torch.zeros(buffer_shape, dtype=data_type, device=device, requires_grad=False)
        self.buffer_len: int = buffer_length
        self.front: int = -1
        self.rear: int = -1

    def reset_and_fill(self, data: torch.Tensor):
        assert data.shape == self.data_shape
        self.front = 0
        self.rear = self.buffer_len - 1
        for i in range(self.buffer_len):
            self.buffer[i] = data.clone().to(self.device)

    def record(self, data: torch.Tensor):
        self.front = (self.front + 1) % self.buffer_len
        self.rear = (self.rear + 1) % self.buffer_len

        self.buffer[self.rear] = data.clone().to(self.device)

    def get_latest_data(self):
        return self.buffer[self.rear].clone()

    def get_index_data(self, indices_from_back):
        assert isinstance(indices_from_back, (torch.Tensor, list, int))
        if isinstance(indices_from_back, torch.Tensor):
            indices = ((self.rear - indices_from_back + self.buffer_len) % self.buffer_len).to(torch.long)
        elif isinstance(indices_from_back, list):
            indices = ((self.rear - torch.tensor(indices_from_back, dtype=torch.long) + self.buffer_len) % self.buffer_len).to(torch.long)
        else:
            indices = int((self.rear - indices_from_back + self.buffer_len) % self.buffer_len)

        return self.buffer[indices].clone()


if __name__ == "__main__":
    buffer = CircleBuffer((3, 2), torch.float32, 5)
    init = torch.Tensor([[1., 1.], [2., 2.], [3., 3.]])
    buffer.reset_and_fill(init)
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
    print(buffer.get_index_data(torch.Tensor([0, 1, 3])))
