import torch
import numpy as np
import time

import sys

sys.path.append("/home/wsh/Documents/ros2Projects/QuadrupedSim_webots_ros2/ControllerWrapper/lib")
import vec_mitcontroller
from vec_mitcontroller import SingleController, VectorizedController


class VecControllerBridge:
    def __init__(self, num_controllers, num_threads=1, device='cuda:0'):
        self.device = device
        self.num_controllers = num_controllers
        self.num_threads = num_threads
        self.dof_order = torch.tensor([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8], dtype=torch.int64, device=self.device,
                                      requires_grad=False)
        self.contact_order = torch.tensor([1, 0, 3, 2], dtype=torch.int64, device=self.device, requires_grad=False)

        self.torque_order = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.int64, device=self.device,
                                         requires_grad=False)
        self.torque_inverse_index = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11], dtype=torch.int64, device=self.device,
                                                 requires_grad=False)

        self.controller = VectorizedController(num_controllers, num_threads,
                                               "/home/wsh/Documents/ros2Projects/QuadrupedSim_webots_ros2/ControllerWrapper/config/")

        self.count = 0

        self.motion_order = torch.arange(start=0, end=56, dtype=torch.int64, device=self.device, requires_grad=False)
        # for i in range(4):
        #     self.motion_order[3 + 4 * i: 7 + 4 * i] = self.contact_order + (3 + 4 * i)
        self.motion_order[3:7] = self.contact_order + 3  # gait_period_offset
        self.motion_order[7:11] = self.contact_order + 7  # gait_duty_cycle_offset
        self.motion_order[11:15] = 11 + torch.tensor([3, 0, 1, 2], dtype=torch.int64, device=self.device, requires_grad=False)  # gait_phase_offset ([FL RR RL FR] -> [FR FL RR RL])
        self.motion_order[15:19] = self.contact_order + 15  # swing_clearance_offset
        self.motion_order[28:40] = self.torque_order + 28  # des_feet_pos_rel_hip
        self.motion_order[40:44] = self.contact_order + 40  # feet_mid_bias_xy->x
        self.motion_order[44:48] = self.contact_order + 44  # feet_mid_bias_xy->y
        self.motion_order[48:52] = self.contact_order + 48  # feet_lift_height_bias->height
        self.motion_order[52:56] = self.contact_order + 52  # feet_lift_height_bias->phase_bias

        self.force_ff_numpy = np.zeros((self.num_controllers, 12), dtype=np.float32)
        self.torque_numpy = np.zeros((self.num_controllers, 12), dtype=np.float32)
        self.tau_ff_numpy = np.zeros((self.num_controllers, 12), dtype=np.float32)
        self.q_des_numpy = np.zeros((self.num_controllers, 12), dtype=np.float32)
        self.qd_des_numpy = np.zeros((self.num_controllers, 12), dtype=np.float32)
        # set ndarray contiguous
        self.force_ff_numpy = np.ascontiguousarray(self.force_ff_numpy)
        self.torque_numpy = np.ascontiguousarray(self.torque_numpy)
        self.tau_ff_numpy = np.ascontiguousarray(self.tau_ff_numpy)
        self.q_des_numpy = np.ascontiguousarray(self.q_des_numpy)
        self.qd_des_numpy = np.ascontiguousarray(self.qd_des_numpy)
        # set ndarray to be writeable
        self.force_ff_numpy.setflags(write=1)
        self.torque_numpy.setflags(write=1)
        self.tau_ff_numpy.setflags(write=1)
        self.q_des_numpy.setflags(write=1)
        self.qd_des_numpy.setflags(write=1)

        self.force_ff = torch.zeros(self.num_controllers, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque = torch.zeros(self.num_controllers, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.tau_ff = torch.zeros(self.num_controllers, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.q_des = torch.zeros(self.num_controllers, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.qd_des = torch.zeros(self.num_controllers, 12, dtype=torch.float, device=self.device, requires_grad=False)

    def step_run(self,
                   reset_buf: torch.Tensor,
                   base_quat: torch.Tensor,
                   base_ang_vel: torch.Tensor,
                   base_lin_acc: torch.Tensor,
                   dof_pos: torch.Tensor,
                   dof_vel: torch.Tensor,
                   contact_state: torch.Tensor,
                   motion_planning_cmd: torch.Tensor):
        t1 = time.time_ns()
        # print("count: ", self.count)

        reset_env_ids_cpu = reset_buf.contiguous().cpu().numpy()
        base_quat_cpu = base_quat.contiguous().cpu().numpy()
        base_ang_vel_cpu = base_ang_vel.contiguous().cpu().numpy()
        base_lin_acc_cpu = base_lin_acc.contiguous().cpu().numpy()
        dof_pos_cpu = dof_pos[:, self.dof_order].contiguous().cpu().numpy()
        dof_vel_cpu = dof_vel[:, self.dof_order].contiguous().cpu().numpy()
        contact_state_cpu = (contact_state[:, self.contact_order] * 0.5).contiguous().cpu().numpy()
        motion_planning_cmd_cpu = motion_planning_cmd[:, self.motion_order].contiguous().cpu().numpy()

        # print("reset_env_ids_cpu: ", reset_env_ids_cpu)
        # print("base_quat_cpu: ", base_quat_cpu)
        # print("base_ang_vel_cpu: ", base_ang_vel_cpu)
        # print("base_lin_acc_cpu: ", base_lin_acc_cpu)
        # print("dof_pos_cpu: ", dof_pos_cpu)
        # print("dof_vel_cpu: ", dof_vel_cpu)
        # print("contact_state_cpu: ", contact_state_cpu)
        # print("motion_planning_cmd_cpu: ", motion_planning_cmd_cpu)
        # print("motion_planning_cmd: ", motion_planning_cmd)

        # reset_env_ids_cpu = reset_buf.contiguous().cpu()
        # base_quat_cpu = base_quat.contiguous().cpu()
        # base_ang_vel_cpu = base_ang_vel.contiguous().cpu()
        # base_lin_acc_cpu = base_lin_acc.contiguous().cpu()
        # dof_pos_cpu = dof_pos[:, self.dof_order].contiguous().cpu()
        # dof_vel_cpu = dof_vel[:, self.dof_order].contiguous().cpu()
        # contact_state_cpu = contact_state[:, self.contact_order].contiguous().cpu()

        dof_pos_cpu[:, 4:] *= -1.
        dof_vel_cpu[:, 4:] *= -1.

        self.controller.getMotorCommands2(self.force_ff_numpy, self.torque_numpy, self.tau_ff_numpy, self.q_des_numpy, self.qd_des_numpy,
                                          reset_env_ids_cpu, base_quat_cpu, base_ang_vel_cpu, base_lin_acc_cpu,
                                          dof_pos_cpu, dof_vel_cpu, contact_state_cpu, motion_planning_cmd_cpu)

        # print("py_torque_numpy", torque_numpy)
        self.force_ff[:] = torch.from_numpy(self.force_ff_numpy).float().to(self.device)[:, self.torque_order]
        self.torque[:] = torch.from_numpy(self.torque_numpy).float().to(self.device)[:, self.torque_order]
        self.torque[:, self.torque_inverse_index] *= -1.
        # print("py_torque", self.torque)
        self.tau_ff[:] = torch.from_numpy(self.tau_ff_numpy).float().to(self.device)[:, self.torque_order]
        self.tau_ff[:, self.torque_inverse_index] *= -1.
        self.q_des[:] = torch.from_numpy(self.q_des_numpy).float().to(self.device)[:, self.torque_order]
        self.q_des[:, self.torque_inverse_index] *= -1.
        self.qd_des[:] = torch.from_numpy(self.qd_des_numpy).float().to(self.device)[:, self.torque_order]
        self.qd_des[:, self.torque_inverse_index] *= -1.

        # torque = torch.zeros_like(dof_pos_cpu)
        # # torque = self.controller.calTorqueTensor(reset_env_ids_cpu, base_quat_cpu, base_ang_vel_cpu, base_lin_acc_cpu, dof_pos_cpu, dof_vel_cpu, contact_state_cpu)
        # self.controller.getMotorCommands(torque, reset_env_ids_cpu, base_quat_cpu, base_ang_vel_cpu, base_lin_acc_cpu, dof_pos_cpu, dof_vel_cpu, contact_state_cpu)
        # torque = torque.to('cuda:0')[:, self.torque_order]
        # torque[self.torque_inverse_index] *= -1.

        # print("torque_numpy[0]", torque_numpy[0])
        t2 = time.time_ns()
        self.count += 1
        # print("convert time1: ", (t2 - t1) / 1.e6, " ms")
        # print("py_torque_numpy", torque_numpy)
        # print("py_torque", torque)
        return self.force_ff.clone(), self.torque.clone(), self.tau_ff.clone(), self.q_des.clone(), self.qd_des.clone()


class SingleControllerBridge:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.dof_order = torch.tensor([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8], dtype=torch.int64, device=self.device,
                                      requires_grad=False)
        self.contact_order = torch.tensor([1, 0, 3, 2], dtype=torch.int64, device=self.device, requires_grad=False)
        self.torque_order = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=torch.int64, device=self.device,
                                         requires_grad=False)
        self.torque_inverse_index = torch.tensor([1, 2, 4, 5, 7, 8, 10, 11], dtype=torch.int64, device=self.device,
                                                 requires_grad=False)

        self.controller = SingleController("/home/wsh/Documents/ros2Projects/QuadrupedSim_webots_ros2/ControllerWrapper/config/")

        self.motion_order = torch.arange(start=0, end=28, dtype=torch.int64, device=self.device, requires_grad=False)
        for i in range(4):
            self.motion_order[3 + 4 * i: 7 + 4 * i] = self.contact_order + (3 + 4 * i)

    def get_torque(self,
                   controller_reset_buf: torch.Tensor,
                   base_quat: torch.Tensor,
                   base_ang_vel: torch.Tensor,
                   base_lin_acc: torch.Tensor,
                   dof_pos: torch.Tensor,
                   dof_vel: torch.Tensor,
                   contact_state: torch.Tensor,
                   motion_planning_cmd: torch.Tensor):
        t1 = time.time_ns()

        reset_env_ids_cpu = controller_reset_buf.contiguous().cpu().numpy()
        base_quat_cpu = base_quat.contiguous().cpu().numpy()
        base_ang_vel_cpu = base_ang_vel.contiguous().cpu().numpy()
        base_lin_acc_cpu = base_lin_acc.contiguous().cpu().numpy()
        dof_pos_cpu = dof_pos[:, self.dof_order].contiguous().cpu().numpy()
        dof_vel_cpu = dof_vel[:, self.dof_order].contiguous().cpu().numpy()
        contact_state_cpu = (contact_state[:, self.contact_order] * 0.5).contiguous().cpu().numpy()
        motion_planning_cmd_cpu = motion_planning_cmd[:, self.motion_order].contiguous().cpu().numpy()

        dof_pos_cpu[:, 4:] *= -1.
        dof_vel_cpu[:, 4:] *= -1.

        # reset_env_ids_cpu = np.zeros((1,), dtype=np.int64)
        # base_quat_cpu = np.zeros((1, 4), dtype=np.float)
        # base_ang_vel_cpu = np.zeros((1, 3), dtype=np.float)
        # base_lin_acc_cpu = np.zeros((1, 3), dtype=np.float)
        # dof_pos_cpu = np.zeros((1, 12), dtype=np.float)
        # dof_vel_cpu = np.zeros((1, 12), dtype=np.float)
        # contact_state_cpu = np.zeros((1, 4), dtype=np.float)

        torque_numpy = np.zeros((1, 12), dtype=np.float32)
        # set torque_numpy contiguous
        torque_numpy = np.ascontiguousarray(torque_numpy)
        # set torque_numpy to be writeable
        torque_numpy.setflags(write=1)
        self.controller.getMotorCommands2(torque_numpy, reset_env_ids_cpu, base_quat_cpu, base_ang_vel_cpu,
                                          base_lin_acc_cpu, dof_pos_cpu, dof_vel_cpu, contact_state_cpu,
                                          motion_planning_cmd_cpu)
        torque = torch.zeros_like(dof_pos)
        torque[:] = torch.from_numpy(torque_numpy).float().to(self.device)[:, self.torque_order]
        torque[:, self.torque_inverse_index] *= -1.

        t2 = time.time_ns()
        print("convert time1: ", (t2 - t1) / 1.e6, " ms")
        print("py_torque_numpy[0]", torque_numpy)
        print("py_torque[0]", torque)

        return torque.clone()
