import torch


class ExponentialAverager:
    def __init__(self, buf_record_length: int, num_buckets: int, alpha: float):
        self.buf_record_length = buf_record_length
        self.alpha = alpha
        self.num_buckets = num_buckets
        self.history_buffer = torch.empty(0, 3)

    def reset(self, ids):

    def update(self, vx, vy, omega_yaw):
        if self.history_buffer.numel() == 0:
            self.history_buffer = torch.tensor([[vx, vy, omega_yaw]])
        else:
            self.history_buffer = torch.cat([self.history_buffer, torch.tensor([[vx, vy, omega_yaw]])])
            if self.history_buffer.shape[0] > self.buf_record_length:
                self.history_buffer = self.history_buffer[-self.buf_record_length:]

    def get_smoothed_values(self):
        smoothed_vx = torch.mean(self.history_buffer[:, 0] * self.alpha + (1 - self.alpha) * self.history_buffer[-1, 0])
        smoothed_vy = torch.mean(self.history_buffer[:, 1] * self.alpha + (1 - self.alpha) * self.history_buffer[-1, 1])
        smoothed_omega_yaw = torch.mean(self.history_buffer[:, 2] * self.alpha + (1 - self.alpha) * self.history_buffer[-1, 2])

        return smoothed_vx, smoothed_vy, smoothed_omega_yaw
