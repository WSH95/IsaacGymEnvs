import torch
from typing import Union


class MotionPlanningInterface:
    def __init__(self, num_robots: int, length: int, device='cuda:0'):
        self.num_robots = num_robots
        self.motion_cmd_length = length
        self.device = device
        self.motion_planning_cmd = torch.zeros(self.num_robots, self.motion_cmd_length, dtype=torch.float,
                                               device=self.device, requires_grad=False)

        self.gait_to_change = torch.zeros(self.num_robots, 1, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.body_state_to_change = torch.zeros(self.num_robots, 1, dtype=torch.float, device=self.device,
                                                requires_grad=False)

        self.body_height_offset = torch.zeros(self.num_robots, 1, dtype=torch.float, device=self.device,
                                              requires_grad=False)
        self.gait_period_offset = torch.zeros(self.num_robots, 4, dtype=torch.float, device=self.device,
                                              requires_grad=False)
        self.gait_duty_cycle_offset = torch.zeros(self.num_robots, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False)
        self.gait_phase_offset = torch.zeros(self.num_robots, 4, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.swing_clearance_offset = torch.zeros(self.num_robots, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False)
        self.body_orientation = torch.zeros(self.num_robots, 3, dtype=torch.float, device=self.device,
                                            requires_grad=False)
        self.body_linear_velocity = torch.zeros(self.num_robots, 3, dtype=torch.float, device=self.device,
                                                requires_grad=False)
        self.body_angular_velocity = torch.zeros(self.num_robots, 3, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.des_feet_pos_rel_hip = torch.zeros(self.num_robots, 12, dtype=torch.float, device=self.device,
                                                 requires_grad=False)

    def get_motion_command(self):
        return self.motion_planning_cmd.clone()

    def generate_motion_command(self):
        self.motion_planning_cmd[:] = torch.cat([self.gait_to_change,
                                                 self.body_state_to_change,
                                                 self.body_height_offset,
                                                 self.gait_period_offset,
                                                 self.gait_duty_cycle_offset,
                                                 self.gait_phase_offset,
                                                 self.swing_clearance_offset,
                                                 self.body_orientation,
                                                 self.body_linear_velocity,
                                                 self.body_angular_velocity,
                                                 self.des_feet_pos_rel_hip],
                                                dim=-1)

    def update_gait_planning(self,
                             gait_to_change: Union[torch.Tensor, bool] = False,
                             gait_period_offset: Union[torch.Tensor, None] = None,
                             gait_duty_cycle_offset: Union[torch.Tensor, None] = None,
                             gait_phase_offset: Union[torch.Tensor, None] = None,  # [FL RR RL FR]
                             swing_clearance_offset: Union[torch.Tensor, None] = None):
        if isinstance(gait_to_change, bool):
            if gait_to_change:
                self.gait_to_change[:] = 1
            else:
                self.gait_to_change[:] = 0
        else:
            self.gait_to_change[:] = gait_to_change.clone()

        if gait_period_offset is not None:
            self.gait_period_offset[:] = gait_period_offset.clone()
        if gait_duty_cycle_offset is not None:
            self.gait_duty_cycle_offset[:] = gait_duty_cycle_offset.clone()
        if gait_phase_offset is not None:
            self.gait_phase_offset[:] = gait_phase_offset.clone()
        if swing_clearance_offset is not None:
            self.swing_clearance_offset[:] = swing_clearance_offset.clone()

    def update_body_planning(self,
                             body_state_to_change: Union[torch.Tensor, bool] = False,
                             body_height_offset: Union[torch.Tensor, None] = None,
                             body_orientation: Union[torch.Tensor, None] = None,
                             body_linear_velocity: Union[torch.Tensor, None] = None,
                             body_angular_velocity: Union[torch.Tensor, None] = None,
                             body_vel_x_y_wz: Union[torch.Tensor, None] = None):
        if isinstance(body_state_to_change, bool):
            if body_state_to_change:
                self.body_state_to_change[:] = 1
            else:
                self.body_state_to_change[:] = 0
        else:
            self.body_state_to_change[:] = body_state_to_change.clone()

        if body_height_offset is not None:
            self.body_height_offset[:] = body_height_offset.clone()
        if body_orientation is not None:
            self.body_orientation[:] = body_orientation.clone()
        if body_linear_velocity is not None:
            self.body_linear_velocity[:] = body_linear_velocity.clone()
        if body_angular_velocity is not None:
            self.body_angular_velocity[:] = body_angular_velocity.clone()
        if body_vel_x_y_wz is not None:
            self.body_linear_velocity[:, :2] = body_vel_x_y_wz[:, :2].clone()
            self.body_linear_velocity[:, 2] = 0
            self.body_angular_velocity[:, :2] = 0
            self.body_angular_velocity[:, 2] = body_vel_x_y_wz[:, 2].clone()

    def update_des_feet_pos_rel_hip(self, des_feet_pos_rel_hip: torch.Tensor):
        self.des_feet_pos_rel_hip = des_feet_pos_rel_hip.clone()

    def change_gait_planning(self, flag: bool):
        if flag:
            self.gait_to_change[:] = 1
            self.motion_planning_cmd[:, 0] = 1
        else:
            self.gait_to_change[:] = 0
            self.motion_planning_cmd[:, 0] = 0

    def change_body_planning(self, flag: bool):
        if flag:
            self.body_state_to_change[:] = 1
            self.motion_planning_cmd[:, 1] = 1
        else:
            self.body_state_to_change[:] = 0
            self.motion_planning_cmd[:, 1] = 0


import time
if __name__ == "__main__":
    num_robots = 6
    length = 28
    device = 'cuda:0'
    motion = MotionPlanningInterface(num_robots, length, device)
    motion_planning_cmd = torch.zeros(num_robots, length, dtype=torch.float, device=device, requires_grad=False)

    gait_to_change = torch.zeros(num_robots, 1, dtype=torch.float, device=device, requires_grad=False)
    body_state_to_change = torch.zeros(num_robots, 1, dtype=torch.float, device=device, requires_grad=False)
    body_height_offset = torch.zeros(num_robots, 1, dtype=torch.float, device=device, requires_grad=False)
    gait_period_offset = torch.zeros(num_robots, 4, dtype=torch.float, device=device, requires_grad=False)
    gait_duty_cycle_offset = torch.zeros(num_robots, 4, dtype=torch.float, device=device, requires_grad=False)
    gait_phase_offset = torch.zeros(num_robots, 4, dtype=torch.float, device=device, requires_grad=False)
    swing_clearance_offset = torch.zeros(num_robots, 4, dtype=torch.float, device=device, requires_grad=False)
    body_orientation = torch.zeros(num_robots, 3, dtype=torch.float, device=device, requires_grad=False)
    body_linear_velocity = torch.zeros(num_robots, 3, dtype=torch.float, device=device, requires_grad=False)
    body_angular_velocity = torch.zeros(num_robots, 3, dtype=torch.float, device=device, requires_grad=False)
    body_vel_x_y_wz = torch.zeros(num_robots, 3, dtype=torch.float, device=device, requires_grad=False)

    t1 = time.time_ns()
    motion.update_gait_planning(gait_to_change, gait_period_offset, gait_duty_cycle_offset, gait_phase_offset, None)
    motion.update_body_planning(body_state_to_change, body_height_offset, body_orientation, None, None, body_vel_x_y_wz)
    motion.generate_motion_command()
    motion_planning_cmd[:] = motion.get_motion_command()
    t2 = time.time_ns()
    print("time1: ", (t2 - t1) / 1.e6, " ms")
    print(motion_planning_cmd)

    gait_to_change = True
    body_state_to_change = True
    gait_period_offset[:] = 0.1
    gait_duty_cycle_offset[:] = 0.2
    gait_phase_offset[:] = 0.3
    body_height_offset[:] = 0.4
    body_orientation[:] = 0.5
    body_vel_x_y_wz[:, 0] = 0.6
    body_vel_x_y_wz[:, 1] = 0.7
    body_vel_x_y_wz[:, 2] = 0.8

    t1 = time.time_ns()
    motion.update_gait_planning(gait_to_change, gait_period_offset, gait_duty_cycle_offset, gait_phase_offset, None)
    motion.update_body_planning(body_state_to_change, body_height_offset, body_orientation, None, None, body_vel_x_y_wz)
    motion.generate_motion_command()
    motion_planning_cmd[:] = motion.get_motion_command()
    t2 = time.time_ns()
    print("time2: ", (t2 - t1) / 1.e6, " ms")
    print(motion_planning_cmd)

    t1 = time.time_ns()
    motion.change_gait_planning(True)
    motion.change_body_planning(False)
    motion_planning_cmd[:] = motion.get_motion_command()
    t2 = time.time_ns()
    print("time3: ", (t2 - t1) / 1.e6, " ms")
    print(motion_planning_cmd)
