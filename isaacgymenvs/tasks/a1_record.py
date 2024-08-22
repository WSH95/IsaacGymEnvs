# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os, time, sys
from tqdm import tqdm
# from isaacgym.torch_utils import *

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask

import torch
from typing import Tuple, Dict

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, normalize, quat_rotate, quat_apply, quat_rotate_inverse, copysign, quat_from_euler_xyz, quat_mul

from isaacgymenvs.utils.circle_buffer import CircleBuffer

from isaacgymenvs.utils.observation_utils import ObservationBuffer

from isaacgymenvs.utils.controller_bridge import SingleControllerBridge, VecControllerBridge

from isaacgymenvs.utils.motion_planning_interface import MotionPlanningInterface

from isaacgymenvs.utils.gait_tracking_policy import GaitTrackingPolicy

from isaacgymenvs.utils.leg_kinematics import QuadrupedLegKinematics2

import random

from torch.distributions import Normal
from typing import Union
import progressbar
from isaacgymenvs.utils.data_description import *

class A1Record(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.init_done = False
        # while True:
        #     print("wait...")

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rpy = self.cfg["env"]["baseInitState"]["rpy"]
        quat = gymapi.Quat.from_euler_zyx(*rpy)
        rot = [quat.x, quat.y, quat.z, quat.w]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang
        self.desired_base_height = 0.3

        # threshold
        self.contact_force_threshold = self.cfg["env"]["contactForceThreshold"]
        self.stance_foot_force_threshold = self.cfg["env"]["stanceFootForceThreshold"]
        xy_velocity_threshold_list = self.cfg["env"]["xywVelocityCommandThreshold"]

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]  ### wsh_annotation: TODO different scale

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"]
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"]
        self.rew_scales["orient"] = self.cfg["env"]["learn"]["orientationRewardScale"]
        self.rew_scales["torques"] = self.cfg["env"]["learn"]["torquesRewardScale"]
        self.rew_scales["delta_torques"] = self.cfg["env"]["learn"]["deltaTorquesRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["knee_collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        self.rew_scales["stumble"] = self.cfg["env"]["learn"]["feetStumbleRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["dof_bias"] = self.cfg["env"]["learn"]["dofBiasRewardScale"]
        self.rew_scales["energy"] = self.cfg["env"]["learn"]["energyRewardScale"]
        self.rew_scales["power"] = self.cfg["env"]["learn"]["powerRewardScale"]
        self.rew_scales["power_max_mean_each"] = self.cfg["env"]["learn"]["power_max_mean_each"]
        self.rew_scales["power_max_mean_std"] = self.cfg["env"]["learn"]["power_max_mean_std"]
        self.rew_scales["feet_max_force_total"] = self.cfg["env"]["learn"]["feet_max_force_total"]
        self.rew_scales["feet_max_force_std"] = self.cfg["env"]["learn"]["feet_max_force_std"]
        self.rew_scales["torque_max_mean_each"] = self.cfg["env"]["learn"]["torque_max_mean_each"]
        self.rew_scales["torque_max_mean_std"] = self.cfg["env"]["learn"]["torque_max_mean_std"]
        self.rew_scales["fallen_over"] = self.cfg["env"]["learn"]["fallenOverRewardScale"]
        self.rew_scales["gait_tracking"] = self.cfg["env"]["learn"]["gaitTrackingScale"]
        self.rew_scales["gait_trans_rate"] =self.cfg["env"]["learn"]["gaitTransRateScale"]
        self.rew_scales["gait_phase_timing"] = self.cfg["env"]["learn"]["gaitPhaseTimingScale"]
        self.rew_scales["gait_phase_shape"] = self.cfg["env"]["learn"]["gaitPhaseShapeScale"]
        self.rew_scales["imitation_torque"] = self.cfg["env"]["learn"]["imitationTorque"]
        self.rew_scales["imitation_joint_pos"] = self.cfg["env"]["learn"]["imitationJointPos"]
        self.rew_scales["imitation_joint_vel"] = self.cfg["env"]["learn"]["imitationJointVel"]
        self.rew_scales["imitation_foot_pos"] = self.cfg["env"]["learn"]["imitationFootPos"]
        self.rew_scales["imitation_foot_vel"] = self.cfg["env"]["learn"]["imitationFootVel"]
        self.rew_scales["feet_contact_regulate"] = self.cfg["env"]["learn"]["feetContactRegulate"]

        self.reward_weights = {"qr": self.cfg["env"]["learn"]["rewards"]["weights"]["qr"],
                            "contact_schedule": self.cfg["env"]["learn"]["rewards"]["weights"]["contact_schedule"],
                            "kine_imitation": self.cfg["env"]["learn"]["rewards"]["weights"]["kine_imitation"],
                            "dyna_imitation": self.cfg["env"]["learn"]["rewards"]["weights"]["dyna_imitation"],
                            "smooth": self.cfg["env"]["learn"]["rewards"]["weights"]["smooth"]}
        self.reward_scales = {"swing_schedule": self.cfg["env"]["learn"]["rewards"]["scales"]["swing_schedule"],
                              "stance_schedule": self.cfg["env"]["learn"]["rewards"]["scales"]["stance_schedule"],
                              "feet_pos_xy": self.cfg["env"]["learn"]["rewards"]["scales"]["feet_pos_xy"],
                              "feet_pos_z": self.cfg["env"]["learn"]["rewards"]["scales"]["feet_pos_z"],
                              "feet_vel_xy": self.cfg["env"]["learn"]["rewards"]["scales"]["feet_vel_xy"],
                              "feet_vel_z": self.cfg["env"]["learn"]["rewards"]["scales"]["feet_vel_z"],
                              "dof_bias": self.cfg["env"]["learn"]["rewards"]["scales"]["dof_bias"],
                              "feet_lin_momentum": self.cfg["env"]["learn"]["rewards"]["scales"]["feet_lin_momentum"],
                              "feet_ang_momentum": self.cfg["env"]["learn"]["rewards"]["scales"]["feet_ang_momentum"],
                              "whole_lin_momentum": self.cfg["env"]["learn"]["rewards"]["scales"]["whole_lin_momentum"],
                              "whole_ang_momentum": self.cfg["env"]["learn"]["rewards"]["scales"]["whole_ang_momentum"],
                              "action_rate": self.cfg["env"]["learn"]["rewards"]["scales"]["action_rate"],
                              "collision": self.cfg["env"]["learn"]["rewards"]["scales"]["collision"],
                              "stumble": self.cfg["env"]["learn"]["rewards"]["scales"]["stumble"]}

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["learn"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["learn"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["learn"]["randomCommandVelocityRanges"]["yaw"]
        self.command_gait_period_range = self.cfg["env"]["learn"]["randomCommandGaitRanges"]["period"]
        self.command_gait_duty_range = self.cfg["env"]["learn"]["randomCommandGaitRanges"]["duty"]
        self.command_gait_offset_range = self.cfg["env"]["learn"]["randomCommandGaitRanges"]["offset"]
        self.command_height_range = self.cfg["env"]["learn"]["randomCommandHeightRanges"]
        self.if_schedule_command = self.cfg["env"]["learn"]["ifScheduleCommand"]

        # push ranges
        self.push_velocity_range = self.cfg["env"]["learn"]["randomPushRanges"]["velocity"]

        # mass random
        self.mass_interval_s = self.cfg["env"]["learn"]["randomMassParams"]["interval_s"]
        self.randomize_base_mass = self.cfg["env"]["learn"]["randomMassParams"]["randomize_base_mass"]
        self.added_mass_range = self.cfg["env"]["learn"]["randomMassParams"]["added_mass_range"]
        self.randomize_base_com = self.cfg["env"]["learn"]["randomMassParams"]["randomize_base_com"]
        self.added_com_range = self.cfg["env"]["learn"]["randomMassParams"]["added_com_range"]

        self.scheduled_command_x_range = self.command_x_range
        self.scheduled_command_y_range = self.command_y_range
        self.scheduled_command_yaw_range = self.command_yaw_range
        self.scheduled_push_velocity_range = self.push_velocity_range
        self.scheduled_command_height_range = self.command_height_range

        # other
        # self.decimation = self.cfg["env"]["control"]["decimation"]
        # self.dt = self.decimation * self.cfg["sim"]["dt"]
        # self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        # self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        # self.push_flag = self.cfg["env"]["learn"]["pushRobots"]
        # self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        kp = self.cfg["env"]["control"]["stiffness"]
        kd = self.cfg["env"]["control"]["damping"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.add_terrain_obs = self.cfg["env"]["terrain"]["addTerrainObservation"]
        self.num_terrain_obs = self.cfg["env"]["terrain"]["numTerrainObservations"]



        # for key in self.rew_scales.keys():
        #     self.rew_scales[key] *= self.dt  ### wsh_annotation: TODO for what???

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.add_terrain_obs:
            self.cfg["env"]["numObservations"] += self.num_terrain_obs
            self.terrain_height = torch.zeros(self.num_envs, self.num_terrain_obs, dtype=torch.float, device=self.device, requires_grad=False)

        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.sim_dt = self.cfg["sim"]["dt"]
        self.dt = self.decimation * self.sim_dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_flag = self.cfg["env"]["learn"]["pushRobots"]

        push_interval = self.cfg["env"]["learn"]["pushInterval_s"]
        commands_change_interval = self.cfg["env"]["learn"]["commands_change_s"]
        gait_commands_change_interval = self.cfg["env"]["learn"]["gait_commands_change_s"]
        height_commands_change_interval = self.cfg["env"]["learn"]["height_commands_change_s"]
        self.push_interval = [int(push_interval[0] / self.dt), int(push_interval[1] / self.dt) + 1]
        self.commands_change_interval = [int(commands_change_interval[0] / self.dt), int(commands_change_interval[1] / self.dt) + 1]
        self.gait_commands_change_interval = [int(gait_commands_change_interval[0] / self.dt), int(gait_commands_change_interval[1] / self.dt) + 1]
        self.height_commands_change_interval = [int(height_commands_change_interval[0] / self.dt), int(height_commands_change_interval[1] / self.dt) + 1]

        self.push_random_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.commands_change_random_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.gait_commands_change_random_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.height_commands_change_random_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.current_push_velocity = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.current_push_velocity[:] = torch_rand_float(self.scheduled_push_velocity_range[0], self.scheduled_push_velocity_range[1], (self.num_envs, 2), device=self.device)

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt  ### wsh_annotation: TODO for what???

        self.has_fallen = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            self.set_camera(p, lookat)
            # cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            # cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            # pose = gymapi.Transform()
            # pose.p = gymapi.Vec3(-1, -1, 0)
            # self.gym.attach_camera_to_body(0, 0, 0, pose, gymapi.CameraFollowMode.FOLLOW_TRANSFORM)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        sensor_forces = self.gym.acquire_force_sensor_tensor(self.sim)
        actor_rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # mass_matrix_tensor = self.gym.acquire_mass_matrix_tensor(self.sim, "a1")

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_mass_matrix_tensors(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_pos_rel_init = torch.zeros_like(self.dof_pos)
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        ### wsh_annotation: use force sensor or not
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)  # shape: num_envs*num_bodies, xyz axis
        self.sensor_forces = gymtorch.wrap_tensor(sensor_forces)
        self.rigid_body_states = gymtorch.wrap_tensor(actor_rigid_body_state)
        self.rigid_body_states_reshape = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        self.feet_position_world = self.rigid_body_states_reshape[:, self.feet_indices, 0:3].view(self.num_envs, -1)
        self.feet_lin_vel_world = self.rigid_body_states_reshape[:, self.feet_indices, 7:10].view(self.num_envs, -1)
        self.feet_position_body = torch.zeros_like(self.feet_position_world)
        self.feet_position_body_in_world_frame = torch.zeros_like(self.feet_position_world)
        self.feet_lin_vel_body = torch.zeros_like(self.feet_lin_vel_world)
        self.feet_lin_vel_body_from_joint = torch.zeros_like(self.feet_lin_vel_world)
        self.feet_position_hip = torch.zeros_like(self.feet_position_body)
        self.feet_position_hip_from_joint = torch.zeros_like(self.feet_position_world)
        self.feet_position_moved_hip = torch.zeros_like(self.feet_position_hip)
        self.feet_position_hip_horizon_frame = torch.zeros_like(self.feet_position_hip)
        self.feet_velocity_hip_horizon_frame = torch.zeros_like(self.feet_position_hip)
        self.feet_position_moved_hip_horizon_frame = torch.zeros_like(self.feet_position_hip_horizon_frame)
        hip_position_rel_body = self.cfg["env"]["urdfAsset"]["hip_position_rel_body"]
        self.hip_position_rel_body = torch.tensor(hip_position_rel_body, dtype=torch.float, device=self.device, requires_grad=False)
        leg_bias_rel_hip = self.cfg["env"]["urdfAsset"]["leg_bias_rel_hip"]
        self.leg_bias_rel_hip = torch.tensor(leg_bias_rel_hip, dtype=torch.float, device=self.device, requires_grad=False)
        self.leg_bias_rel_hip_xy = self.leg_bias_rel_hip[[0, 1, 3, 4, 6, 7, 9, 10]].reshape(4, 2)
        self.feet_position_hip[:] = self.feet_position_body - self.hip_position_rel_body
        self.feet_position_moved_hip[:] = self.feet_position_hip - self.leg_bias_rel_hip
        self.link_length = self.cfg["env"]["urdfAsset"]["link_length"]
        self.body_half_length = self.cfg["env"]["urdfAsset"]["body_half_length"]
        self.robot_mass = self.cfg["env"]["urdfAsset"]["robot_mass"]
        self.robot_weight = 9.8 * self.robot_mass

        self.feet_height_rel_ground = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # self.mass_matrix = gymtorch.wrap_tensor(mass_matrix_tensor)

        self.Kp = torch.zeros_like(self.dof_pos)
        self.Kd = torch.zeros_like(self.dof_vel)
        self.Kp[:, :] = kp
        self.Kd[:, :] = kd
        # self.Kp[:, [0, 3, 6, 9]] = 25.0
        # self.Kd[:, [0, 3, 6, 9]] = 1.0


        if self.cfg["env"]["urdfAsset"]["useForceSensor"]:
            self.contact_forces = self.sensor_forces[:, :3].view(self.num_envs, -1,
                                                                 3)  # shape: num_envs, num_bodies, xyz axis
        else:
            self.contact_forces = self.net_contact_forces.view(self.num_envs, -1,
                                                               3)  # shape: num_envs, num_bodies, xyz axis
        self.feet_force = self.contact_forces[:, self.feet_indices].view(self.num_envs, -1)

        # initialize some data used later on
        self.common_step_counter = 0
        self.simulate_counter = 0
        # self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) ### wsh_annotation
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                    requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_last = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)  # x vel, y vel, yaw vel
        self.command_lin_vel_x = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                    requires_grad=False)
        self.vel_commands_body = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.vel_commands_world = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_lin_vel_horizon_feet = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.Rw_matrix = torch.zeros(self.num_envs, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.delta_theta = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel_command = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_ang_vel_command = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_heading_flag = torch.ones(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.commands_delta = 1.e4 * self.dt
        self.schedule_delta = 1.
        self.gait_commands = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device,
                                    requires_grad=False)  # period, duty, offset2, offset3, offset4, phase
        self.gait_commands_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.gait_params_act = torch.zeros_like(self.gait_commands)
        self.gait_params_act_last = torch.zeros_like(self.gait_commands)
        self.gait_params_act_raw_last = torch.zeros(self.num_envs, 7, dtype=torch.float, device=self.device,
                                                    requires_grad=False)
        self.gait_periods = torch.zeros_like(self.gait_commands[:, 0])
        self.gait_period_act = torch.zeros_like(self.gait_commands[:, 0])
        self.gait_period_tracking_error_last = torch.zeros_like(self.gait_commands[:, 0])
        self.phase_overwrite = torch.zeros_like(self.gait_commands[:, 3])
        self.phase_overwrite_last = torch.zeros_like(self.phase_overwrite)

        self.gait_duty = torch.zeros_like(self.gait_commands[:, 1])
        self.gait_stance_time = torch.zeros_like(self.gait_commands[:, 0])

        self.ref_phase_sincos_current = torch.zeros(self.num_envs, 8, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_phase_pi_current = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_delta_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_delta_phase_sim_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_phase_norm_sincos_current = torch.zeros_like(self.ref_phase_sincos_current)
        self.ref_phase_norm_pi_current = torch.zeros_like(self.ref_phase_pi_current)
        self.ref_phase_norm_sincos_next = torch.zeros_like(self.ref_phase_sincos_current)
        self.ref_phase_norm_pi_next = torch.zeros_like(self.ref_phase_pi_current)
        self.ref_phase_current = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_phase_norm_current = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        sigma = self.cfg["env"]["learn"]["refPhaseTransDistribution"]
        self.ref_phase_trans_distribution = Normal(torch.zeros_like(self.ref_phase_norm_current), sigma)
        self.ref_phase_C_des = torch.zeros_like(self.ref_phase_norm_current)
        self.foot_pos_track_weight = torch.zeros_like(self.ref_phase_norm_current)

        self.ref_foot_pos_xy_horizon = torch.zeros(self.num_envs, 4, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_foot_vel_xy_horizon = torch.zeros(self.num_envs, 4, 2, dtype=torch.float, device=self.device, requires_grad=False)

        self.feet_phase_sincos = torch.zeros_like(self.ref_phase_sincos_current)

        self.ground_height_current = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.height_commands = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                           requires_grad=False)
        self.body_orientation_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                                           device=self.device, requires_grad=False, )
        self.xy_velocity_threshold = torch.tensor(xy_velocity_threshold_list, dtype=torch.float, device=self.device, requires_grad=False)

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_torques = torch.zeros_like(self.torques)
        self.ref_torques = torch.zeros_like(self.torques)
        self.ref_tau_ff = torch.zeros_like(self.torques)
        self.ref_dof_pos = torch.zeros_like(self.torques)
        self.ref_dof_vel = torch.zeros_like(self.torques)
        self.action_tau_ff = torch.zeros_like(self.torques)
        self.action_dof_pos = torch.zeros_like(self.torques)
        self.action_dof_vel = torch.zeros_like(self.torques)

        self.force_ff_mpc = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.est_feet_force = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_feet_force_mpc = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_lin_momentum = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_ang_momentum = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_feet_lin_momentum = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_feet_ang_momentum = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)

        self.ref_body_trajectory = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.act_body_trajectory = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        self.body_traj_error = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        body_traj_tracking_weight = self.cfg["env"]["learn"]["rewards"]["scales"]["bodyTrajTrackingWeight"]
        self.body_traj_tracking_weight = torch.tensor(body_traj_tracking_weight, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_weight = self.cfg["env"]["learn"]["rewards"]["scales"]["torqueWeight"]
        self.init_position_bias_rel_world = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_actions_raw = torch.zeros_like(self.actions)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.feet_contact_state = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) ### wsh_annotation: 1->contact
        self.feet_contact_state_obs = torch.zeros_like(self.feet_contact_state)

        self.ref_phase_contact_state = torch.zeros_like(self.feet_contact_state)
        self.ref_phase_contact_num = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_feet_force = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)

        self.height_points = self.init_height_points()
        self.measured_heights = None
        # joint positions offsets
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device,
                                                requires_grad=False)

        # set motor broken
        self.motor_broken_count = torch.randint(0, 2, (self.num_envs,), device=self.device, dtype=torch.long)
        self.motor_broken_count[:] = 1
        self.motor_broken_table = torch.zeros(self.num_envs, 3, dtype=torch.long, device=self.device, requires_grad=False)
        env_id_count0 = torch.where(self.motor_broken_count == 0)[0]
        env_id_count1 = torch.where(self.motor_broken_count == 1)[0]
        env_id_count2 = torch.where(self.motor_broken_count == 2)[0]
        env_id_count_not0 = torch.where(self.motor_broken_count != 0)[0]
        self.motor_broken_table[env_id_count0, :] = -1
        self.motor_broken_table[env_id_count1, 1:] = -2
        self.motor_broken_table[env_id_count2, 2] = -torch.randint(3, 5, (len(env_id_count2),), device=self.device, dtype=torch.long)
        self.motor_broken_table[env_id_count_not0, 0] = torch.randint(0, self.num_dof, (len(env_id_count_not0),), device=self.device, dtype=torch.long)
        # self.motor_broken_table[env_id_count_not0, 0] = 0
        self.motor_broken_table[env_id_count2, 1] = torch.randint(0, self.num_dof, (len(env_id_count2),), device=self.device, dtype=torch.long)
        env_id_count_2_duplicate = torch.where((self.motor_broken_table[:, 0] == self.motor_broken_table[:, 1]) & (self.motor_broken_table[:, 2] < -2))[0]
        self.motor_broken_table[env_id_count_2_duplicate, 1] = (self.motor_broken_table[env_id_count_2_duplicate, 1] + 1) % self.num_dof
        self.motor_not_broken_flag = torch.ones(self.num_envs, self.num_dof, dtype=torch.long, device=self.device, requires_grad=False)
        self.motor_not_broken_flag1 = torch.ones(self.num_envs, self.num_dof, dtype=torch.long, device=self.device, requires_grad=False)
        self.motor_not_broken_flag2 = torch.ones(self.num_envs, self.num_dof, dtype=torch.long, device=self.device, requires_grad=False)
        self.leg_broken_flag = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device, requires_grad=False)
        self.leg_not_broken_flag = (~self.leg_broken_flag.bool()).to(torch.long)
        self.leg_broken_flag1 = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device, requires_grad=False)
        self.leg_broken_flag2 = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device, requires_grad=False)
        self.motor_not_broken_flag1[env_id_count_not0, self.motor_broken_table[env_id_count_not0, 0]] = 0
        env_id_count2_3 = torch.where(self.motor_broken_table[:, 2] == -3)[0]
        self.motor_not_broken_flag1[env_id_count2_3, self.motor_broken_table[env_id_count2_3, 1]] = 0
        self.motor_not_broken_flag2 = self.motor_not_broken_flag1.clone()
        env_id_count2_4 = torch.where(self.motor_broken_table[:, 2] == -4)[0]
        self.motor_not_broken_flag2[env_id_count2_4, self.motor_broken_table[env_id_count2_4, 1]] = 0
        # self.motor_not_broken_flag2[:] = 1
        # self.motor_not_broken_flag2[:, [7, 10]] = 0
        self.leg_broken_flag1 = torch.any(self.motor_not_broken_flag1.resize(self.num_envs, 4, 3) == 0, dim=2).to(torch.long)
        self.leg_broken_flag2 = torch.any(self.motor_not_broken_flag2.resize(self.num_envs, 4, 3) == 0, dim=2).to(torch.long)
        self.motor_broken_count1 = torch.randint(250, 251, (self.num_envs,), dtype=torch.long, device=self.device)
        self.motor_broken_count2 = torch.randint(950, 1050, (self.num_envs,), dtype=torch.long, device=self.device)
        self.leg_broken_count = torch.sum(self.leg_broken_flag, dim=1)
        self.gait_list_leg_broken = [[0.3, 0.75, 0.5, 0.0, 0.75, 0.25, 1.0, 1.0, 1.0, 1.0],
                                     [0.3, 2.0 / 3.0, 5.0 / 6.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 0.0, 1.0, 1.0, 1.0],
                                     [0.3, 2.0 / 3.0, 0.0, 5.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 0.0, 1.0, 1.0],
                                     [0.3, 2.0 / 3.0, 0.0, 1.0 / 3.0, 5.0 / 6.0, 2.0 / 3.0, 1.0, 1.0, 0.0, 1.0],
                                     [0.3, 2.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 5.0 / 6.0, 1.0, 1.0, 1.0, 0.0],
                                     [0.3, 0.5, 0.75, 0.75, 0.0, 0.5, 0.0, 0.0, 1.0, 1.0],
                                     [0.3, 0.5, 0.75, 0.0, 0.75, 0.5, 0.0, 1.0, 0.0, 1.0],
                                     [0.3, 0.5, 0.75, 0.0, 0.0, 0.75, 0.0, 1.0, 1.0, 0.0],
                                     [0.3, 0.5, 0.0, 0.75, 0.75, 0.0, 1.0, 0.0, 0.0, 1.0],
                                     [0.3, 0.5, 0.0, 0.75, 0.5, 0.75, 1.0, 0.0, 1.0, 0.0],
                                     [0.3, 0.5, 0.0, 0.5, 0.75, 0.75, 1.0, 1.0, 0.0, 0.0],
                                     [0.3, 0.5, 0.75, 0.75, 0.75, 0.0, 0.0, 0.0, 0.0, 1.0],
                                     [0.3, 0.5, 0.75, 0.75, 0.0, 0.75, 0.0, 0.0, 1.0, 0.0],
                                     [0.3, 0.5, 0.75, 0.0, 0.75, 0.75, 0.0, 1.0, 0.0, 0.0],
                                     [0.3, 0.5, 0.0, 0.75, 0.75, 0.75, 1.0, 0.0, 0.0, 0.0]]
        self.gait_tensor_leg_broken = torch.tensor(self.gait_list_leg_broken, dtype=torch.float, device=self.device, requires_grad=False)
        self.index_leg_broken_flag_to_gait = torch.zeros(2, 2, 2, 2, dtype=torch.long, device=self.device, requires_grad=False)
        self.index_leg_broken_flag_to_gait[0, 0, 0, 0] = 0
        self.index_leg_broken_flag_to_gait[1, 0, 0, 0] = 1
        self.index_leg_broken_flag_to_gait[0, 1, 0, 0] = 2
        self.index_leg_broken_flag_to_gait[0, 0, 1, 0] = 3
        self.index_leg_broken_flag_to_gait[0, 0, 0, 1] = 4
        self.index_leg_broken_flag_to_gait[1, 1, 0, 0] = 5
        self.index_leg_broken_flag_to_gait[1, 0, 1, 0] = 6
        self.index_leg_broken_flag_to_gait[1, 0, 0, 1] = 7
        self.index_leg_broken_flag_to_gait[0, 1, 1, 0] = 8
        self.index_leg_broken_flag_to_gait[0, 1, 0, 1] = 9
        self.index_leg_broken_flag_to_gait[0, 0, 1, 1] = 10
        self.index_leg_broken_flag_to_gait[1, 1, 1, 0] = 11
        self.index_leg_broken_flag_to_gait[1, 1, 0, 1] = 12
        self.index_leg_broken_flag_to_gait[1, 0, 1, 1] = 13
        self.index_leg_broken_flag_to_gait[0, 1, 1, 1] = 14
        self.index_leg_broken_flag_to_gait[1, 1, 1, 1] = 15
        self.gait_index_leg_broken = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.gait_params_leg_broken = torch.zeros(self.num_envs, 10, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_phase_leg_broken = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_phase_normed_leg_broken = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_phase_C_des_leg_broken = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_phase_norm_sincos_leg_broken = torch.zeros(self.num_envs, 8, dtype=torch.float, device=self.device, requires_grad=False)

        self.global_clock_period = 0.3
        self.global_clock_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.global_clock_phase_pi = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.global_clock_phase_sin_cos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(len(self.dof_names)):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle
        # reward episode sums
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "lin_vel_z": torch_zeros(), "ang_vel_z": torch_zeros(),
                             "ang_vel_xy": torch_zeros(),
                             "orient": torch_zeros(), "torques": torch_zeros(), "delta_torques": torch_zeros(),
                             "joint_acc": torch_zeros(), "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "knee_collision": torch_zeros(), "stumble": torch_zeros(),
                             "action_rate": torch_zeros(), "energy": torch_zeros(), "power": torch_zeros(),
                             "hip": torch_zeros(), "dof_bias": torch_zeros(),
                             "power_max_mean_each": torch_zeros(), "power_max_mean_std": torch_zeros(),
                             "feet_max_force_total": torch_zeros(), "feet_max_force_std": torch_zeros(),
                             "torque_max_mean_each": torch_zeros(), "torque_max_mean_std": torch_zeros(),
                             "fallen_over": torch_zeros(), "gait_tracking": torch_zeros(),
                             "gait_trans_rate": torch_zeros(), "gait_phase_timing": torch_zeros(),
                             "gait_phase_shape": torch_zeros(), "imitation_torque": torch_zeros(),
                             "imitation_joint_pos": torch_zeros(), "imitation_joint_vel": torch_zeros(),
                             "imitation_foot_pos": torch_zeros(), "imitation_foot_vel": torch_zeros(),
                             "feet_contact_regulate": torch_zeros(), "feet_height_leg_broken": torch_zeros(),
                             "fault_joint_vel": torch_zeros()}

        self.episode_sums2 = {
            "qr": torch_zeros(), "contact_schedule": torch_zeros(), "kine_imitation": torch_zeros(), "dyna_imitation": torch_zeros(), "smooth": torch_zeros(),
            "traj_pos_xy": torch_zeros(), "traj_pos_z": torch_zeros(), "traj_ang_xy": torch_zeros(),
            "traj_ang_z": torch_zeros(), "traj_lin_vel_xy": torch_zeros(), "traj_lin_vel_z": torch_zeros(),
            "traj_ang_vel_xy": torch_zeros(), "traj_ang_vel_z": torch_zeros(), "traj_sum": torch_zeros(), "torque": torch_zeros(),
            "swing_schedule": torch_zeros(), "stance_schedule": torch_zeros(),
            "feet_pos_xy": torch_zeros(), "feet_pos_z": torch_zeros(), "feet_vel_xy": torch_zeros(), "feet_vel_z": torch_zeros(), "dof_bias": torch_zeros(),
            "feet_lin_momentum": torch_zeros(), "feet_ang_momentum": torch_zeros(), "whole_lin_momentum": torch_zeros(), "whole_ang_momentum": torch_zeros(),
            "action_rate": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros()
        }

        self.rew_error = {
            "traj_pos_xy": torch_zeros(), "traj_pos_z": torch_zeros(), "traj_ang_xy": torch_zeros(),
            "traj_ang_z": torch_zeros(), "traj_lin_vel_xy": torch_zeros(), "traj_lin_vel_z": torch_zeros(),
            "traj_ang_vel_xy": torch_zeros(), "traj_ang_vel_z": torch_zeros(), "traj_sum": torch_zeros(),
            "torque": torch_zeros(),
            "swing_schedule": torch_zeros(), "stance_schedule": torch_zeros(),
            "feet_pos_xy": torch_zeros(), "feet_pos_z": torch_zeros(), "feet_vel_xy": torch_zeros(),
            "feet_vel_z": torch_zeros(), "dof_bias": torch_zeros(),
            "feet_lin_momentum": torch_zeros(), "feet_ang_momentum": torch_zeros(), "whole_lin_momentum": torch_zeros(),
            "whole_ang_momentum": torch_zeros(),
            "action_rate": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros()
        }

        self.base_quat = self.root_states[:, 3:7]
        self.euler_xyz = get_euler_xyz2(self.base_quat)
        self.world2base_quat = torch.zeros_like(self.base_quat)
        self.horizon_quat_in_world = torch.zeros_like(self.base_quat)
        self.horizon_quat_in_base = torch.zeros_like(self.base_quat)
        self.base_quat_horizon = torch.zeros_like(self.base_quat)

        self.euler_roll = self.euler_xyz.view(self.num_envs, 1, 3)[..., 0]
        self.euler_pitch = self.euler_xyz.view(self.num_envs, 1, 3)[..., 1]
        self.euler_yaw = self.euler_xyz.view(self.num_envs, 1, 3)[..., 2]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.horizon_lin_vel = torch.zeros_like(self.base_lin_vel)
        self.horizon_ang_vel = torch.zeros_like(self.base_ang_vel)

        # acceleration
        self.last_base_lin_vel_rel_world = self.root_states[:, 7:10].clone().detach()
        self.gravity_acc = torch.tensor([0., 0., -9.81], dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_acc = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                        requires_grad=False) - self.gravity_acc

        # extra force or torque
        self.extra_force = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device,
                                       requires_grad=False)
        self.extra_torque = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device,
                                        requires_grad=False)

        # controller reset buf
        self.controller_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        # motion planning command
        # self.motion_planning_cmd = torch.zeros(self.num_envs, 40, dtype=torch.float, device=self.device, requires_grad=False)

        self.power_norm = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.vx_mean = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # accumulated square state
        # self.torques_square_accumulated = torch.zeros_like(self.torques)
        # self.base_lin_vel_error_square_accumulated = torch.zeros_like(self.base_lin_vel)
        # self.base_ang_vel_error_square_accumulated = torch.zeros_like(self.base_ang_vel)

        self.mixed_actions_raw = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.mixed_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)

        ### wsh_annotation: observations dict
        self.obs_name_to_value = {"linearVelocity": self.base_lin_vel,
                                  "angularVelocity": self.base_ang_vel,
                                  "projectedGravity": self.projected_gravity,
                                  "dofPosition": self.dof_pos,
                                  "dofVelocity": self.dof_vel,
                                  "lastAction": self.last_actions_raw,
                                  "commands": self.commands[:, :3],
                                  "feetContactState": self.feet_contact_state_obs,
                                  "bodyPos": self.root_states[:, :3],
                                  "motorTorque": self.torques,
                                  "feetForce": self.feet_force,
                                  "dofPositionRelInit": self.dof_pos_rel_init,
                                  "rollAngle": self.euler_roll,
                                  "pitchAngle": self.euler_pitch,
                                  "gaitCommands": self.gait_commands,
                                  "heightCommands": self.height_commands,
                                  "feetPositionRelHip": self.feet_position_moved_hip,
                                  "feetLinVelRelHip": self.feet_lin_vel_body,
                                  "gaitParamsAct": self.gait_params_act,
                                  "armature_coeffs_real": self.armature_coeffs_real,
                                  "friction_coeffs_real": self.friction_coeffs_real,
                                  "power_norm": self.power_norm,
                                  "command_lin_vel_x": self.command_lin_vel_x,
                                  "vx_mean": self.vx_mean,
                                  "ref_phase_current": self.ref_phase_current,
                                  "feet_phase_sincos": self.feet_phase_sincos,
                                  "ref_phase_norm_sincos_current": self.ref_phase_norm_sincos_current,
                                  "ref_phase_norm_sincos_next": self.ref_phase_norm_sincos_next,
                                  "body_traj_err": self.body_traj_error,
                                  "motor_not_broken_flag": self.motor_not_broken_flag,
                                  "leg_not_broken_flag": self.leg_not_broken_flag,
                                  "ref_phase_norm_sincos_leg_broken": self.ref_phase_norm_sincos_leg_broken,
                                  "gait_phase_leg_broken": self.gait_phase_leg_broken,
                                  "global_clock_phase_sin_cos": self.global_clock_phase_sin_cos}

        self.obs_combination = self.cfg["env"]["learn"]["observationConfig"]["combination"]
        self.states_combination = self.cfg["env"]["learn"]["observationConfig"]["states_combination"]
        print("observations:")
        for key in self.obs_combination.keys():
            print(key)
        print("states:")
        for key in self.states_combination.keys():
            print(key)
        self.obs_components = self.cfg["env"]["learn"]["observationConfig"]["components"]
        add_obs_noise = self.cfg["env"]["learn"]["observationConfig"]["addNoise"]
        self.obs_buffer_dict = {}
        self.record_items = self.obs_components.keys()
        if self.add_terrain_obs:
            self.obs_name_to_value["heightMeasurement"] = self.terrain_height
        elif "heightMeasurement" in self.record_items:
            keys_set = set(self.record_items)
            keys_set.remove("heightMeasurement")
            self.record_items = self.record_items & keys_set
        key_set_duration_sim_dt = set(self.record_items)
        key_set_duration_learn_dt = set(self.record_items)
        for key in self.record_items:
            if add_obs_noise:
                noise = self.obs_components[key]["noise"]
            else:
                noise = None
            if self.obs_components[key]["size"]:
                self.obs_buffer_dict[key] = ObservationBuffer(num_envs=self.num_envs,
                                                              single_data_shape=(self.obs_components[key]["size"],),
                                                              data_type=torch.float,
                                                              buffer_length=self.obs_components[key]["bufferLength"],
                                                              device=self.device,
                                                              scale=self.obs_components[key]["scale"],
                                                              noise=noise)
            if not self.obs_components[key]["during_sim_dt"]:
                key_set_duration_sim_dt.remove(key)
            else:
                key_set_duration_learn_dt.remove(key)
        self.record_items_duration_sim_dt = self.record_items & key_set_duration_sim_dt
        self.record_items_duration_learn_dt = self.record_items & key_set_duration_learn_dt

        # self.len_obs_history = self.cfg["env"]["lenObsHis"]
        # assert self.num_obs % self.len_obs_history == 0
        # self.num_obs_single = self.num_obs // self.len_obs_history
        # self.obs_single = torch.zeros(self.num_envs, self.num_obs_single, dtype=torch.float, device=self.device)
        # self.obs_circle_buffer = CircleBuffer(self.num_envs, (self.num_obs_single,), torch.float, self.len_obs_history, self.device)

        self.vel_average = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)

        self.motion_planning_interface = MotionPlanningInterface(self.num_envs, 56, self.device)
        self.gait_period_offset = torch.zeros_like(self.feet_contact_state)
        self.gait_duty_cycle_offset = torch.zeros_like(self.feet_contact_state)
        self.gait_phase_offset = torch.zeros_like(self.feet_contact_state)
        self.des_feet_pos_rel_hip = torch.zeros_like(self.feet_position_body)
        self.feet_mid_bias_xy = torch.zeros_like(self.feet_position_body[:, :8])
        self.feet_lift_height_bias = torch.zeros_like(self.feet_mid_bias_xy)

        self.dof_order_act = torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=torch.int64, device=self.device,
                                          requires_grad=False)
        self.dof_order_obs = torch.tensor([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=torch.int64, device=self.device,
                                          requires_grad=False)
        self.dof_order_obs = torch.arange(12, dtype=torch.int64, device=self.device, requires_grad=False)
        self.dof_order_act = torch.arange(12, dtype=torch.int64, device=self.device, requires_grad=False)

        self.side_coef = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.side_coef[:, :2] = 1
        self.side_coef[:, 2:] = -1

        # kinematics
        self.robot_kinematics = QuadrupedLegKinematics2(self.num_envs, self.link_length[0], self.link_length[1], self.link_length[2], self.device)
        self.jacobian = torch.zeros(self.num_envs, 4, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.inverse_jacobian = torch.zeros(self.num_envs, 4, 3, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.kinematic_feet_pos = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.gait_list = [[0.5, 0.75, 0.5, 0.25, 0.75, 0.],  # 0
                          [0.3, 0.75, 0.5, 0.25, 0.75, 0.],  # 1
                          [0.5, 0.5, 0.5, 0.5, 0., 0.],  # 2
                          [0.3, 0.5, 0.5, 0.5, 0., 0.],  # 3
                          [0.5, 0.3, 0.5, 0.5, 0., 0.],  # 4
                          [0.3, 0.3, 0.5, 0.5, 0., 0.],  # 5
                          [0.25, 0.6, 0.5, 0., 0.5, 0.],  # 6
                          [0.3, 0.5, 0.5, 0., 0.5, 0.],  # 7
                          [0.3, 0.3, 0., 0., 0., 0.],  # 8
                          [0.3, 0.5, 0., 0., 0., 0.],  # 9
                          [0.3, 0.5, 0., 0.5, 0.5, 0.5],  # 10
                          [1.0, 0.5, 0.5, 0.5, 0., 0.]]  # 11
        self.gait_tensor = torch.tensor(self.gait_list, device=self.device)
        self.num_gait = len(self.gait_list)
        self.gait_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        self.leg_broken_policy = GaitTrackingPolicy(12, 68, [512, 256, 128])
        self.leg_broken_policy.to(self.device)
        self.leg_broken_policy.eval()
        self.leg_broken_policy.restore_from_file(
            "/home/wsh/Documents/pyProjects/IsaacGymEnvs/isaacgymenvs/runs/A1Limited_2024-05-26_03-38-19(obs+flag+height+noGait_Terrain)/nn/A1Limited.pth")
        # self.leg_broken_policy.restore_from_file(
        #     "/home/wsh/Documents/pyProjects/IsaacGymEnvs/isaacgymenvs/runs/A1Limited_2024-05-12_07-07-26/nn/last_A1Limited_ep_2000_rew_35.40136.pth")
        self.leg_broken_policy_actions = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                                     requires_grad=False)
        self.leg_broken_policy_actions_raw = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device,
                                                         requires_grad=False)
        self.extras['ref_actions'] = self.leg_broken_policy_actions_raw

        self.env_step_height = (self.terrain_levels + 1) * 0.005 + 0.05

        self.schedule_random()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # self.compute_observations()
        self.init_done = True

        self.mit_controller = VecControllerBridge(self.num_envs, self.cfg["num_controller_threads"], self.device)
        # self.motion_planning_interface = MotionPlanningInterface(self.num_envs, 28, self.device)

        # lower gait tracking layer
        # self.gait_tracking_policy = GaitTrackingPolicy(7, 77, [512, 256, 128])
        # self.gait_tracking_policy.to(self.device)
        # self.gait_tracking_policy.eval()
        # self.normalize_gait_tracking_input = True
        # self.restore_gait_tracking_policy("/home/wsh/Documents/pyProjects/IsaacGymEnvs/isaacgymenvs/runs/A1_2023-10-19_18-12-18(gait_transition_2)/nn/A1.pth")

        self.record_data = np.expand_dims(np.arange(57), axis=0)
        self.record_data_test = np.expand_dims(np.arange(116+self.num_actions), axis=0)
        self.record_path = ''
        self.time_step = time.time()

        # sample for vel tracking simulation
        # for ii in range(106):
        #     for jj in range(106):
        #         self.commands[ii * 106 + jj, 0] = ii * 0.02 - 0.55
        #         self.commands[ii * 106 + jj, 1] = 0
        #         self.commands[ii * 106 + jj, 2] = jj * 0.02 - 1.05

        self.motor_not_broken_flag1[:] = 1
        vx_list = [-0.5, 0, 0.5, 1.0, 1.5, 0]
        vy_list = [0, 0, 0, 0, 0, 0.5]
        for ii in range(6):
            self.commands[ii*1200: (ii+1)*1200, 0] = vx_list[ii]
            self.commands[ii*1200: (ii+1)*1200, 1] = vy_list[ii]
            for jj in range(12):
                self.motor_not_broken_flag1[ii*1200+jj*100: ii*1200+(jj+1)*100, jj] = 0
        # self.motor_not_broken_flag1[:] = 1
        self.leg_broken_flag1 = torch.any(self.motor_not_broken_flag1.resize(self.num_envs, 4, 3) == 0, dim=2).to(torch.long)

        self.commands[:, 0] = 0.5
        self.commands[:, 1] = 0.


    def create_sim(self):
        if self.cfg["sim"]["up_axis"] == "z":
            self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        else:
            self.up_axis_idx = 1
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        terrain_type = self.cfg["env"]["terrain"]["terrainType"]
        if terrain_type == 'plane':
            self._create_ground_plane()
        elif terrain_type == 'trimesh':
            self._create_trimesh()
            self.custom_origins = True
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg["env"]["learn"]["addNoise"]
        noise_level = self.cfg["env"]["learn"]["noiseLevel"]
        noise_vec[:3] = self.cfg["env"]["learn"]["linearVelocityNoise"] * noise_level * self.lin_vel_scale
        noise_vec[3:6] = self.cfg["env"]["learn"]["angularVelocityNoise"] * noise_level * self.ang_vel_scale
        noise_vec[6:9] = self.cfg["env"]["learn"]["gravityNoise"] * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = self.cfg["env"]["learn"]["dofPositionNoise"] * noise_level * self.dof_pos_scale
        noise_vec[24:36] = self.cfg["env"]["learn"]["dofVelocityNoise"] * noise_level * self.dof_vel_scale
        noise_vec[36:176] = self.cfg["env"]["learn"]["heightMeasurementNoise"] * noise_level * self.height_meas_scale
        noise_vec[176:188] = 0.  # previous actions
        return noise_vec

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        t1 = time.time()
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'), tm_params)
        t2 = time.time()
        print(f"time3: {t2 - t1} s")
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"][
            "collapseFixedJoints"]  ### wsh_annotation: modofy 'True' to cfg parameter
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 1000.
        asset_options.max_linear_velocity = 1000.
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        # asset_options.vhacd_enabled = True

        a1_asset = self.gym.load_asset(self.sim, asset_root, asset_file,
                                       asset_options)  ### wsh_annotation: (FL, FR, RL, RR)
        self.num_dof = self.gym.get_asset_dof_count(a1_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(a1_asset)

        # get link name
        body_names = self.gym.get_asset_rigid_body_names(a1_asset)
        self.dof_names = self.gym.get_asset_dof_names(a1_asset)
        base_name = self.cfg["env"]["urdfAsset"]["baseName"]
        thigh_name = self.cfg["env"]["urdfAsset"]["thighName"]
        calf_name = self.cfg["env"]["urdfAsset"]["calfName"]
        foot_name = self.cfg["env"]["urdfAsset"]["footName"]
        # if self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]:
        #     foot_name = calf_name
        thigh_names = [s for s in body_names if thigh_name in s]
        calf_names = [s for s in body_names if calf_name in s]
        feet_names = [s for s in body_names if foot_name in s]

        penalize_contacts_names = [base_name]
        penalize_contacts_names += thigh_names
        penalize_contacts_names += calf_names

        self_collision_names = calf_names + feet_names
        self_collision_ids = [self.gym.find_asset_rigid_body_index(a1_asset, name) for name in self_collision_names]

        num_buckets = 100
        bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))

        # get assert props
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(a1_asset)
        dof_props = self.gym.get_asset_dof_properties(a1_asset)

        armature_range = self.cfg["env"]["learn"]["armatureRange"]
        armature_buckets = torch_rand_float(armature_range[0], armature_range[1], (num_buckets, 1), device=self.device)
        self.armature_coeffs = armature_buckets[bucket_ids]
        self.armature_coeffs_real[:] = self.armature_coeffs.squeeze().unsqueeze(-1)

        # prepare friction & restitution randomization
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        restitution_range = self.cfg["env"]["learn"]["restitutionRange"]
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device=self.device)
        self.friction_coeffs = friction_buckets[bucket_ids]
        restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets, 1),
                                               device=self.device)
        self.friction_coeffs_real[:] = ((self.friction_coeffs.squeeze() + 1.) / 2.).unsqueeze(-1)

        added_mass_buckets = torch_rand_float(self.added_mass_range[0], self.added_mass_range[1], (num_buckets, 1), device=self.device)
        self.added_mass = added_mass_buckets[bucket_ids][:, 0, :]
        added_com_buckets = torch_rand_float(self.added_com_range[0], self.added_com_range[1], (num_buckets, 3), device=self.device)
        self.added_com = added_com_buckets[bucket_ids][:, 0, :]

        # for s in range(len(rigid_shape_prop)):
        #     # rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
        #     # rigid_shape_prop[s].restitution = restitution_buckets[i % num_buckets]
        #     rigid_shape_prop[s].friction = self.cfg["env"]["terrain"]["staticFriction"]
        #     rigid_shape_prop[s].rolling_friction = 0.
        #     rigid_shape_prop[s].torsion_friction = 0.
        #     rigid_shape_prop[s].restitution = self.cfg["env"]["terrain"]["restitution"]
        #     rigid_shape_prop[s].contact_offset = 0.001
        #     rigid_shape_prop[s].rest_offset = -0.003
        # self.gym.set_asset_rigid_shape_properties(a1_asset, rigid_shape_prop)

        # init start pose of robot
        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # wsh_annotation: add force sensor
        sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = True
        sensor_props.enable_constraint_solver_forces = True
        sensor_props.use_world_frame = True
        for i in range(self.num_bodies):
            self.gym.create_asset_force_sensor(a1_asset, i, sensor_pose, sensor_props)

        # env origins
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"][
                                                                                    "numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,),
                                            device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,),
                                           device=self.device)
        self.env_index_plane = torch.where(self.terrain_types > -1.0)[0].to(torch.long)
        self.target_points = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.custom_origins:
            self.env_index_plane = torch.where(self.terrain_types < 2.0)[0].to(torch.long)
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.target_points_list = torch.from_numpy(self.terrain.env_target_points).to(self.device).to(torch.float)
            self.reached_target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
            self.target_pos_rel = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.target_threshold = self.cfg["env"]["terrain"]["targetThreshold"]
            self.target_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            spacing = 0.  ### wsh_annotation: the same origin (0., 0.) of env coordinates

        # prepare for env creatation
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.a1_handles = []
        self.envs = []

        # env creating
        for i in tqdm(range(self.num_envs)):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            ## wsh_annotation: set friction and restitution of robots
            for s in range(len(rigid_shape_prop)):
                rigid_shape_prop[s].friction = self.friction_coeffs[i]
                rigid_shape_prop[s].friction = 0.6
            #     # rigid_shape_prop[s].restitution = restitution_buckets[i % num_buckets]
            #     rigid_shape_prop[s].friction = 1.
            #     rigid_shape_prop[s].rolling_friction = 0.
            #     rigid_shape_prop[s].torsion_friction = 0.
            #     rigid_shape_prop[s].restitution = self.cfg["env"]["terrain"]["restitution"]
            #     rigid_shape_prop[s].contact_offset = 0.001
            #     rigid_shape_prop[s].rest_offset = -0.003
            #     rigid_shape_prop[s].filter = 1
            # for s in self_collision_ids:
            #     rigid_shape_prop[s].filter = 0

            for j in range(self.num_dof):
                dof_props['driveMode'][j] = gymapi.DOF_MODE_EFFORT  # gymapi.DOF_MODE_POS
                # dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
                # dof_props['damping'][j] = 0.01 #self.Kd
                dof_props['armature'][j] = self.armature_coeffs[i]  # 0.01
                # dof_props['armature'][j] = 0.01

            self.gym.set_asset_rigid_shape_properties(a1_asset, rigid_shape_prop)
            a1_handle = self.gym.create_actor(env_handle, a1_asset, start_pose, "a1", i, 1, 0)
            self.gym.set_asset_rigid_shape_properties(a1_asset, rigid_shape_prop)
            self.gym.set_actor_dof_properties(env_handle, a1_handle, dof_props)
            # tmp_dof_props = self.gym.get_actor_dof_properties(env_handle, a1_handle)
            # print(tmp_dof_props)

            rigid_body_prop = self.gym.get_actor_rigid_body_properties(env_handle, a1_handle)
            if self.randomize_base_mass:
                rigid_body_prop[0].mass += self.added_mass[i]
            if self.randomize_base_com:
                rigid_body_prop[0].com += gymapi.Vec3(self.added_com[i][0], self.added_com[i][1], self.added_com[i][2])
            if self.randomize_base_mass or self.randomize_base_com:
                # rigid_body_prop[0].mass = 10.
                # rigid_body_prop[0].com = gymapi.Vec3(0.05, 0.0, 0.0)
                self.gym.set_actor_rigid_body_properties(env_handle, a1_handle, rigid_body_prop, recomputeInertia=True)

            self.envs.append(env_handle)
            self.a1_handles.append(a1_handle)

        # acquire link indices
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device,
                                         requires_grad=False)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0
        self.penalize_contacts_indices = torch.zeros(len(penalize_contacts_names), dtype=torch.long,
                                                     device=self.device, requires_grad=False)
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], thigh_names[i])
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], feet_names[i])
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], base_name)
        for i in range(len(penalize_contacts_names)):
            self.penalize_contacts_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.a1_handles[0], penalize_contacts_names[i])

        print("envs created!")

    def _heading_to_omega(self, heading):
        return torch.clip(0.8 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)
    #
    # def check_termination(self):
    #     self.reset_buf[:] = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
    #     if not self.allow_knee_contacts:
    #         knee_contact = torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1.
    #         self.reset_buf[:] |= torch.any(knee_contact, dim=1)
    #
    #     # pos limit termination
    #     # self.reset_buf[:] |= self.root_states[:, 2] < 0.28
    #     # self.reset_buf[:] |= torch.abs(self.euler_xyz[:, 0]) > 0.2
    #     # self.reset_buf[:] |= torch.abs(self.euler_xyz[:, 1]) > 0.4
    #
    #     self.timeout_buf[:] = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf == 0)
    #     self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf),
    #                                  self.reset_buf)
    #     # print(knee_contact)

    def check_termination(self):
        self.timeout_buf[:] = torch.where(self.progress_buf > self.max_episode_length - 1,
                                          torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))
        self.has_fallen = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.

        if not self.allow_knee_contacts:
            knee_contact = torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1.
            self.has_fallen |= torch.any(knee_contact, dim=1)

        # pos limit termination
        roll_over = torch.abs(self.euler_xyz[:, 0]) > 1.
        pitch_over = torch.abs(self.euler_xyz[:, 1]) > 1.

        self.reset_buf[:] = self.has_fallen.clone()
        self.reset_buf[:] = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf[:] |= roll_over
        self.reset_buf[:] |= pitch_over

    def compute_observations(self):  ### TODO(completed) wsh_annotation: add history buffer and delay. contain terrain info or not.


        # ### wsh_annotation: record new observations into buffer
        # for key in self.record_items:
        #     self.obs_buffer_dict[key].record(self.obs_name_to_value[key])

        # self.commands[:, 0] = 1.
        # self.commands[:, 1] = 0.0
        # self.commands[:, 2] = 0.0

        self.modify_vel_command2()

        # self.commands[:, 0] = 1.0

        self.update_global_clock()
        # self.record_global_clock_phase()
        self.update_motor_broken_state()
        self.update_timing_phase_leg_broken()

        self.modify_desired_gait_command()
        self.calculate_ref_timing_phase()
        # self.record_ref_phase()

        self.modify_desired_height_command()

        # self.record_commands()

        self.update_ref_body_trajectory()

        if self.add_terrain_obs:
            self.measured_heights = self.get_heights()
            self.terrain_height[:] = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)
            # self.obs_buffer_dict["heightMeasurement"].record(self.terrain_height)

        self.record_states_duration_learn_dt()

        # a = self.obs_buffer_dict["gait_phase_leg_broken"].get_index_data([2, 1, 0])
        # b = self.obs_buffer_dict["ref_phase_current"].get_index_data([2, 1, 0])
        # tmp_ids = torch.where((torch.abs((self.ref_phase_norm_current-self.gait_phase_normed_leg_broken))%1.0 > 1.0e-2).any(dim=-1))
        # if len(tmp_ids[0]) > 0:
        #     print(f"broken: {self.gait_phase_leg_broken[0]}")
        #     print(f"custom: {self.ref_phase_current[0]}")
            # print("---------------------------------------------------------------------")
        # print(f"ref_phase_norm_sincos_current: {a}\nref_phase_norm_sincos_legbrok: {b}")
        # print(f"a-b: {(torch.abs(self.ref_phase_norm_current-self.gait_phase_normed_leg_broken) < 1.0e-2).all()}")
        # print(".................")
        # print(f"C_des_n: {self.ref_phase_C_des[5]}\nC_des_b: {self.ref_phase_C_des_leg_broken[5]}")
        # print(f"C_err: {(self.ref_phase_C_des - self.ref_phase_C_des_leg_broken<1.0e-1).all()}")
        # print("---------------------------------------------------------------------")

        # self.obs_single[:] = torch.cat([self.obs_buffer_dict[key].get_index_data(self.obs_combination[key]) for key in self.obs_combination.keys()], dim=-1)
        # init_env_ids = torch.where(self.progress_buf == 0)[0]
        # if len(init_env_ids) > 0:
        #     self.obs_circle_buffer.reset_and_fill_index(init_env_ids, self.obs_single[init_env_ids])
        # self.obs_circle_buffer.record(self.obs_single)
        # self.obs_buf[:] = self.obs_circle_buffer.get_len_data_flatten(self.len_obs_history)
        obs_buf = torch.cat([self.obs_buffer_dict[key].get_index_data(self.obs_combination[key]) for key in self.obs_combination.keys()], dim=-1)
        obs_prop_buf = torch.cat([self.obs_buffer_dict[key].get_latest_data() for key in self.obs_combination.keys()], dim=-1)
        # self.obs_buf[:] = torch.cat([obs_prop_buf, obs_buf[:, 3:]], dim=-1)
        self.obs_buf[:] = obs_buf  # obs_prop_buf  obs_buf

        self.states_buf[:] = torch.cat([self.obs_buffer_dict[key].get_index_data_scaled(self.states_combination[key]) for key in self.states_combination.keys()], dim=-1)
        self.states_buf[:, 52:64] = self.mixed_actions_raw.clone()

        ### wsh_annotation: DEBUG for observation buffer
        # if self.obs_buffer_dict["linearVelocity"].get_latest_data_raw().equal(self.base_lin_vel) and \
        #         self.obs_buffer_dict["angularVelocity"].get_latest_data_raw().equal(self.base_ang_vel) and \
        #         self.obs_buffer_dict["projectedGravity"].get_latest_data_raw().equal(self.projected_gravity) and \
        #         self.obs_buffer_dict["dofPosition"].get_latest_data_raw().equal(self.dof_pos) and \
        #         self.obs_buffer_dict["dofVelocity"].get_latest_data_raw().equal(self.dof_vel) and \
        #         self.obs_buffer_dict["lastAction"].get_latest_data_raw().equal(self.actions) and \
        #         self.obs_buffer_dict["commands"].get_latest_data_raw().equal(self.commands[:, :3]) and \
        #         self.obs_buffer_dict["feetContactState"].get_latest_data_raw().equal(self.feet_contact_state):
        #     print("[success] observation buffer works properly")
        # else:
        #     print("[failed] observation buffer works error")

        # if self.add_terrain_obs:
        #     self.measured_heights = self.get_heights()
        #     heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
        #                          1.) * self.height_meas_scale
        #     self.obs_buf[:] = torch.cat((tmp_obs_buf, heights), dim=-1)
        # else:
        #     self.obs_buf[:] = tmp_obs_buf[:]
        # self.obs_buf1 = self.obs_buf.clone()
        # self.obs_buf = torch.cat((self.base_lin_vel * self.lin_vel_scale,
        #                           self.base_ang_vel * self.ang_vel_scale,
        #                           self.projected_gravity,
        #                           self.commands[:, :3] * self.commands_scale,
        #                           self.dof_pos * self.dof_pos_scale,
        #                           self.dof_vel * self.dof_vel_scale,
        #                           # heights,
        #                           self.actions
        #                           ), dim=-1)
        # self.obs_buf_flag = (self.obs_buf==self.obs_buf1)[0]
        # print(self.obs_buf_flag)
        # print(self.obs_buf1[0, :9])
        # print(self.obs_buf[:, :])

    def compute_reward2(self):
    # qr reward
        # traj tracking reward
        err_traj = torch.square(self.body_traj_error) * self.body_traj_tracking_weight.unsqueeze(0)
        err_traj_pos_xy = torch.sum(err_traj[:, :2], dim=1)
        err_traj_pos_z = err_traj[:, 2]
        err_traj_ang_xy = torch.sum(err_traj[:, 3:5], dim=1)
        err_traj_ang_z = err_traj[:, 5]
        err_traj_lin_vel_xy = torch.sum(err_traj[:, 6:8], dim=1)
        err_traj_lin_vel_z = err_traj[:, 8]
        err_traj_ang_vel_xy = torch.sum(err_traj[:, 9:11], dim=1)
        err_traj_ang_vel_z = err_traj[:, 11]
        err_traj_sum = torch.sum(err_traj, dim=1)
        rew_traj_pos_xy = -err_traj_pos_xy
        rew_traj_pos_z = -err_traj_pos_z
        rew_traj_ang_xy = -err_traj_ang_xy
        rew_traj_ang_z = -err_traj_ang_z
        # rew_traj_lin_vel_xy = -err_traj_lin_vel_xy
        rew_traj_lin_vel_xy = torch.exp(-err_traj_lin_vel_xy / 0.25) * 1.5
        rew_traj_lin_vel_z = -err_traj_lin_vel_z
        rew_traj_ang_vel_xy = -err_traj_ang_vel_xy
        # rew_traj_ang_vel_z = -err_traj_ang_vel_z
        rew_traj_ang_vel_z = torch.exp(-err_traj_ang_vel_z / 0.25) * 1.0
        rew_traj_sum = rew_traj_pos_xy + rew_traj_pos_z + rew_traj_ang_xy + rew_traj_ang_z + rew_traj_lin_vel_xy + rew_traj_lin_vel_z + rew_traj_ang_vel_xy + rew_traj_ang_vel_z
        # torque reward
        err_torque = torch.sum(torch.square(self.torques), dim=1)
        rew_torque = -err_torque * self.torque_weight
        # qr reward
        rew_qr = (rew_traj_sum + rew_torque) * self.reward_weights["qr"]
        # print(f"err_traj_lin_vel_xy: {err_traj_lin_vel_xy[0]}")

    # contact schedule reward
        # swing schedule reward
        feet_force_norm = torch.norm(self.feet_force.view(self.num_envs, 4, 3), p=2, dim=-1)
        err_swing_schedule = torch.square(feet_force_norm)
        # err_swing_schedule = torch.where((0.1 < err_swing_schedule) & (err_swing_schedule < 5.0), 5.0, err_swing_schedule)
        rew_swing_schedule = torch.sum((1. - self.ref_phase_C_des) * torch.exp(-err_swing_schedule * self.reward_scales["swing_schedule"]), dim=1) / 4.0
        # stance schedule reward
        feet_vel_xy_world_norm = torch.norm(self.feet_lin_vel_world.view(self.num_envs, 4, 3)[..., :2], p=2, dim=-1)
        err_stance_schedule = torch.square(feet_vel_xy_world_norm)
        rew_stance_schedule = torch.sum(self.ref_phase_C_des * torch.exp(-err_stance_schedule * self.reward_scales["stance_schedule"]), dim=1) / 4.0
        # contact schedule reward
        rew_contact_schedule = (rew_swing_schedule + rew_stance_schedule) * self.reward_weights["contact_schedule"]
        # rew_contact_schedule = -torch.sum((err_swing_schedule*self.leg_broken_flag > 0.01).to(torch.float), dim=1)*1.0

    # kinematic imitation reward
        # feet pos tracking reward
        tmp_err_feet_pos_xy_each = torch.norm(self.feet_position_moved_hip_horizon_frame.view(self.num_envs, 4, 3)[..., :2] - self.ref_foot_pos_xy_horizon, p=2, dim=-1)
        err_feet_pos_xy = (self.gait_commands_count > 1) * torch.sum(self.foot_pos_track_weight * torch.square(tmp_err_feet_pos_xy_each), dim=-1)
        rew_feet_pos_xy = torch.exp(-err_feet_pos_xy * self.reward_scales["feet_pos_xy"])
        # rew_feet_pos_xy = -err_feet_pos_xy * self.reward_scales["feet_pos_xy"]
        err_feet_pos_z = torch.zeros_like(err_feet_pos_xy)
        rew_feet_pos_z = torch.zeros_like(rew_feet_pos_xy)
        # feet vel tracking reward
        tmp_err_feet_vel_xy_each = torch.norm(self.feet_velocity_hip_horizon_frame.view(self.num_envs, 4, 3)[..., :2] - self.ref_foot_vel_xy_horizon, p=2, dim=-1)
        err_feet_vel_xy = (self.gait_commands_count > 1) * torch.sum(self.foot_pos_track_weight * torch.square(tmp_err_feet_vel_xy_each), dim=-1)
        rew_feet_vel_xy = torch.exp(-err_feet_vel_xy * self.reward_scales["feet_vel_xy"])
        # rew_feet_vel_xy = -err_feet_vel_xy * self.reward_scales["feet_vel_xy"]
        err_feet_vel_z = torch.zeros_like(err_feet_vel_xy)
        rew_feet_vel_z = torch.zeros_like(rew_feet_vel_xy)
        # dof bias reward
        err_dof_bias = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        rew_dof_bias = torch.exp(-err_dof_bias * self.reward_scales["dof_bias"])
        # rew_dof_bias = -err_dof_bias * self.reward_scales["dof_bias"]
        # kinematic imitation reward
        # rew_kine_imitation = (rew_feet_pos_xy * rew_feet_vel_xy * rew_dof_bias - 1.0) * self.reward_weights["kine_imitation"]
        rew_kine_imitation = (rew_feet_pos_xy + rew_feet_vel_xy + rew_dof_bias) * self.reward_weights["kine_imitation"]
        rew_kine_imitation = err_dof_bias * (-0.04)

    # dynamic imitation reward
        tmp_err_feet_lin_momentum_each = self.ref_feet_lin_momentum - self.feet_lin_momentum
        tmp_err_feet_ang_momentum_each = self.ref_feet_ang_momentum - self.feet_ang_momentum
        # feet lin momentum reward
        err_feet_lin_momentum = torch.sum(torch.square(tmp_err_feet_lin_momentum_each), dim=1)
        rew_feet_lin_momentum = torch.exp(-err_feet_lin_momentum * self.reward_scales["feet_lin_momentum"])
        # feet ang momentum reward
        err_feet_ang_momentum = torch.sum(torch.square(tmp_err_feet_ang_momentum_each), dim=1)
        rew_feet_ang_momentum = torch.exp(-err_feet_ang_momentum * self.reward_scales["feet_ang_momentum"])
        # whole lin momentum reward
        err_whole_lin_momentum = torch.sum(torch.square(torch.sum(tmp_err_feet_lin_momentum_each.reshape(self.num_envs, 4, 3), dim=1)), dim=1)
        rew_whole_lin_momentum = torch.exp(-err_whole_lin_momentum * self.reward_scales["whole_lin_momentum"])
        # whole ang momentum reward
        err_whole_ang_momentum = torch.sum(torch.square(torch.sum(tmp_err_feet_ang_momentum_each.reshape(self.num_envs, 4, 3), dim=1)), dim=1)
        rew_whole_ang_momentum = torch.exp(-err_whole_ang_momentum * self.reward_scales["whole_ang_momentum"])
        # dynamic imitation reward
        rew_dyna_imitation = (rew_feet_lin_momentum * rew_feet_ang_momentum * rew_whole_lin_momentum * rew_whole_ang_momentum - 1.0) * self.reward_weights["dyna_imitation"]

    # smoothness reward
        # action rate reward
        err_action_rate = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        # rew_action_rate = torch.exp(-err_action_rate * self.reward_scales["action_rate"])
        rew_action_rate = -err_action_rate * self.reward_scales["action_rate"]
        # collision reward
        knee_contact = torch.norm(self.contact_forces[:, self.penalize_contacts_indices, :], dim=2) > self.contact_force_threshold
        err_knee_collision = torch.sum(knee_contact, dim=1)
        # rew_collision = torch.exp(-err_knee_collision * self.reward_scales["collision"])
        rew_collision = -err_knee_collision * self.reward_scales["collision"]
        # stumble reward
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.) * (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < self.contact_force_threshold)
        err_stumble = torch.any(stumble, dim=1).float()
        # rew_stumble = torch.exp(-err_stumble * self.reward_scales["stumble"])
        rew_stumble = -err_stumble * self.reward_scales["stumble"]
        # smoothness reward
        # rew_smooth = (rew_action_rate * rew_collision * rew_stumble - 1.0) * self.reward_weights["smooth"]
        rew_smooth = (rew_action_rate + rew_collision + rew_stumble) * self.reward_weights["smooth"]

    # calculate total rewards
        self.rew_buf[:] = (rew_qr + rew_contact_schedule + rew_kine_imitation + rew_dyna_imitation + rew_smooth) * self.dt
        self.rew_buf[:] = torch.clip(self.rew_buf, min=0., max=None)

    # sum up episode rewards
        self.episode_sums2["qr"] += rew_qr
        self.episode_sums2["contact_schedule"] += rew_contact_schedule
        self.episode_sums2["kine_imitation"] += rew_kine_imitation
        self.episode_sums2["dyna_imitation"] += rew_dyna_imitation
        self.episode_sums2["smooth"] += rew_smooth

        self.episode_sums2["traj_pos_xy"] += rew_traj_pos_xy
        self.episode_sums2["traj_pos_z"] += rew_traj_pos_z
        self.episode_sums2["traj_ang_xy"] += rew_traj_ang_xy
        self.episode_sums2["traj_ang_z"] += rew_traj_ang_z
        self.episode_sums2["traj_lin_vel_xy"] += rew_traj_lin_vel_xy
        self.episode_sums2["traj_lin_vel_z"] += rew_traj_lin_vel_z
        self.episode_sums2["traj_ang_vel_xy"] += rew_traj_ang_vel_xy
        self.episode_sums2["traj_ang_vel_z"] += rew_traj_ang_vel_z
        self.episode_sums2["traj_sum"] += rew_traj_sum
        self.episode_sums2["torque"] += rew_torque

        self.episode_sums2["swing_schedule"] += rew_swing_schedule
        self.episode_sums2["stance_schedule"] += rew_stance_schedule

        self.episode_sums2["feet_pos_xy"] += rew_feet_pos_xy
        self.episode_sums2["feet_pos_z"] += rew_feet_pos_z
        self.episode_sums2["feet_vel_xy"] += rew_feet_vel_xy
        self.episode_sums2["feet_vel_z"] += rew_feet_vel_z
        self.episode_sums2["dof_bias"] += rew_dof_bias

        self.episode_sums2["feet_lin_momentum"] += rew_feet_lin_momentum
        self.episode_sums2["feet_ang_momentum"] += rew_feet_ang_momentum
        self.episode_sums2["whole_lin_momentum"] += rew_whole_lin_momentum
        self.episode_sums2["whole_ang_momentum"] += rew_whole_ang_momentum

        self.episode_sums2["action_rate"] += rew_action_rate
        self.episode_sums2["collision"] += rew_collision
        self.episode_sums2["stumble"] += rew_stumble

    # record reward errors
        self.rew_error["traj_pos_xy"] += err_traj_pos_xy
        self.rew_error["traj_pos_z"] += err_traj_pos_z
        self.rew_error["traj_ang_xy"] += err_traj_ang_xy
        self.rew_error["traj_ang_z"] += err_traj_ang_z
        self.rew_error["traj_lin_vel_xy"] += err_traj_lin_vel_xy
        self.rew_error["traj_lin_vel_z"] += err_traj_lin_vel_z
        self.rew_error["traj_ang_vel_xy"] += err_traj_ang_vel_xy
        self.rew_error["traj_ang_vel_z"] += err_traj_ang_vel_z
        self.rew_error["traj_sum"] += err_traj_sum
        self.rew_error["torque"] += err_torque

        self.rew_error["swing_schedule"] += -rew_swing_schedule
        self.rew_error["stance_schedule"] += -rew_stance_schedule

        self.rew_error["feet_pos_xy"] += err_feet_pos_xy
        self.rew_error["feet_pos_z"] += err_feet_pos_z
        self.rew_error["feet_vel_xy"] += err_feet_vel_xy
        self.rew_error["feet_vel_z"] += err_feet_vel_z
        self.rew_error["dof_bias"] += err_dof_bias

        self.rew_error["feet_lin_momentum"] += err_feet_lin_momentum
        self.rew_error["feet_ang_momentum"] += err_feet_ang_momentum
        self.rew_error["whole_lin_momentum"] += err_whole_lin_momentum
        self.rew_error["whole_ang_momentum"] += err_whole_ang_momentum

        self.rew_error["action_rate"] += err_action_rate
        self.rew_error["collision"] += err_knee_collision
        self.rew_error["stumble"] += err_stumble

    def compute_reward(self):
        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # lin_vel_error = torch.sum(self.base_lin_vel_error_square_accumulated[:, :2], dim=1) / self.decimation
        tmp_lin_vel_error = self.commands[:, :2] - self.base_lin_vel[:, :2]
        tmp_lin_vel_error = torch.where(torch.abs(tmp_lin_vel_error) < torch.abs(self.commands[:, :2]) * 0.1, torch.zeros_like(tmp_lin_vel_error), tmp_lin_vel_error)
        # tmp_lin_vel_error = torch.where(self.commands[:, :2] * tmp_lin_vel_error < 0, torch.zeros_like(tmp_lin_vel_error), tmp_lin_vel_error)
        # lin_vel_error = torch.sum(torch.square(tmp_lin_vel_error), dim=1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        # rew_lin_vel_xy = torch.minimum(self.base_lin_vel[:, 0], self.commands[:, 0]) / (self.commands[:, 0] + 1e-5) * self.rew_scales["lin_vel_xy"]
        # rew_lin_vel_xy = torch.where((self.commands[:, :2] * tmp_lin_vel_error <= 0).all(dim=-1), rew_lin_vel_xy+0.5, rew_lin_vel_xy)

        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        tmp_ang_vel_error = self.commands[:, 2] - self.base_ang_vel[:, 2]
        tmp_ang_vel_error = torch.where(torch.abs(tmp_ang_vel_error) < torch.abs(self.commands[:, 2]) * 0.1, torch.zeros_like(tmp_ang_vel_error), tmp_ang_vel_error)
        # ang_vel_error = torch.square(tmp_ang_vel_error)
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]
        # rew_ang_vel_z = torch.exp(-torch.abs(self.target_yaw - self.euler_xyz[:, 2])) * self.rew_scales["ang_vel_z"]

        # other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        tmp_base_ang_vel = self.base_ang_vel.clone()
        env_ids_bound = (self.gait_id == 10).nonzero(as_tuple=False).flatten()
        tmp_base_ang_vel[env_ids_bound, 1] *= 0.1
        rew_ang_vel_xy = torch.sum(torch.square(tmp_base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]
        # rew_lin_vel_z = self.base_lin_vel_error_square_accumulated[:, 2] / self.decimation * self.rew_scales["lin_vel_z"]
        # rew_ang_vel_xy = torch.sum(self.base_ang_vel_error_square_accumulated[:, :2], dim=1) / self.decimation * self.rew_scales["ang_vel_xy"]
        # self.base_lin_vel_error_square_accumulated[:] = 0
        # self.base_ang_vel_error_square_accumulated[:] = 0

        # orientation penalty TODO relating to velocity
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]
        # tmp_ori = self.euler_xyz.clone()
        # env_ids_bound = (self.gait_id == 10).nonzero(as_tuple=False).flatten()
        # tmp_ori[env_ids_bound, 1] *= 0.1
        # rew_orient = torch.sum(torch.square(tmp_ori[:, :2] / torch.pi * 180.0), dim=1) * self.rew_scales["orient"]

        # base height penalty
        rew_base_height = torch.square(10.0 * (self.root_states[:, 2] - self.height_commands.squeeze())) * self.rew_scales["base_height"]  # TODO(completed) add target base height to cfg

        # base_height_error = torch.square((self.root_states[:, 2] - self.desired_base_height) * 1000)
        # rew_base_height = (1. - torch.exp(-base_height_error / 10.)) * self.rew_scales["base_height"]
        # print(self.root_states[0, 2])

        # torque penalty TODO power (torque * motor_speed)
        rew_torques = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torques"]
        # rew_torque = torch.sum(self.torques_square_accumulated, dim=1) / self.decimation * self.rew_scales["torques"]
        # self.torques_square_accumulated[:] = 0.0

        rew_delta_torques = torch.sum(torch.square(self.torques - self.last_torques), dim=1) * self.rew_scales["delta_torques"]

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1) * self.rew_scales["joint_acc"]

        # knee_collision penalty
        knee_contact = torch.norm(self.contact_forces[:, self.penalize_contacts_indices, :], dim=2) > self.contact_force_threshold
        rew_knee_collision = torch.sum(knee_contact, dim=1) * self.rew_scales["knee_collision"]  # sum vs any ?

        # fallen over penalty
        rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]
        # action_over = torch.clamp(torch.abs(self.last_actions_raw) - 4., min=0., max=None)
        # rew_fallen_over = torch.norm(action_over, dim=1) * self.rew_scales["fallen_over"]

        # stumbling penalty TODO contact forces x & y are inaccurate
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.) * (
                    torch.abs(self.contact_forces[:, self.feet_indices, 2]) < self.contact_force_threshold)
        rew_stumble = torch.any(stumble, dim=1).float() * self.rew_scales["stumble"]
        # rew_stumble = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
        #                         4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1).float() * self.rew_scales["stumble"]

        # action rate penalty TODO action is uncertain
        rew_action_rate = torch.norm(self.last_actions - self.actions, dim=1) * self.rew_scales["action_rate"]
        # rew_action_rate = torch.norm((self.last_actions[:, 12:] + self.last_actions[:, :12] / 20.0) - (self.actions[:, 12:] + self.actions[:, :12] / 20.0), dim=1) * self.rew_scales["action_rate"]
        # rew_action_rate += torch.norm(self.last_actions[:, :12] - self.actions[:, :12], dim=1) * (-1.0e-3)*self.dt
        # print(f"rew_action_rate: {rew_action_rate[0]}")

        # gait
        # rew_gait_tracking = torch.exp(-torch.norm(self.gait_commands[:, :5] - self.gait_params_act[:, :5], dim=1) / 0.25) * self.rew_scales["gait_tracking"]
        period_tracking_error_current = torch.abs(self.gait_commands[:, 0] - self.gait_params_act[:, 0])
        rew_gait_tracking = torch.exp(-10 * period_tracking_error_current - 4 * torch.norm(self.gait_commands[:, 1:5] - self.gait_params_act[:, 1:5], dim=1))
        rew_gait_tracking -= torch.where(self.gait_period_tracking_error_last < 0.01, 1 - torch.exp(-1000.0 * torch.square(period_tracking_error_current)), torch.zeros_like(period_tracking_error_current))
        rew_gait_tracking *= self.rew_scales["gait_tracking"]
        self.gait_period_tracking_error_last[:] = period_tracking_error_current
        gait_trans = torch.norm(self.gait_params_act[:, :5] - self.gait_params_act_last[:, :5], dim=1)
        rew_gait_trans_rate = gait_trans * self.rew_scales["gait_trans_rate"]
        gait_phase_trans = self.gait_params_act[:, 5] - self.gait_params_act_last[:, 5]
        gait_phase_timing = torch.where(gait_phase_trans < 0, gait_phase_trans + 1, gait_phase_trans) * self.gait_period_act.squeeze()
        rew_gait_phase_timing = torch.where(gait_trans < 0.1, torch.exp(-torch.square((gait_phase_timing - self.dt) / self.dt)), torch.zeros_like(gait_trans))
        rew_gait_phase_timing *= self.rew_scales["gait_phase_timing"]
        rew_gait_phase_shape = (1 - torch.exp(-10 * torch.abs(torch.norm(self.actions[:, 5:7], dim=1) - 1))) * self.rew_scales["gait_phase_shape"]

        # rew_gait_phase_timing = (2.0 - torch.norm(self.ref_phase_sincos_current[:, :2] - self.actions, dim=1)) * self.rew_scales["gait_phase_timing"]
        # phase_error = self.calculate_phase(self.ref_phase_sincos_current[:, 0], self.ref_phase_sincos_current[:, 1]) - \
        #               self.calculate_phase(self.actions[:, 0], self.actions[:, 1])
        # rew_gait_phase_timing += (1.0 - torch.abs(phase_error)) * self.rew_scales["gait_phase_timing"]

        # gait_phase_trans = self.phase_overwrite - self.phase_overwrite_last
        # gait_phase_trans = torch.where(gait_phase_trans < 0, gait_phase_trans + 1, gait_phase_trans)
        # gait_phase_timing = gait_phase_trans * self.gait_periods
        # rew_gait_phase_timing = torch.exp(-torch.abs(gait_phase_timing - self.dt) / self.dt) * self.rew_scales["gait_phase_timing"]
        # self.phase_overwrite_last = self.phase_overwrite

        # air time reward
        # contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1.
        # self.feet_contact_state[:] = self.contact_forces[:, self.feet_indices, 2] > self.stance_foot_force_threshold
        first_contact = (self.feet_air_time > 0.) * self.feet_contact_state
        self.feet_air_time += self.dt
        # reward only on first contact with the ground TODO self.feet_air_time - 0.5 ?
        rew_air_time = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) * self.rew_scales["air_time"]
        # rew_air_time[:] *= torch.norm(self.commands[:, :2], dim=1) > self.xy_velocity_threshold  # no reward for zero command
        rew_air_time[:] *= ~((self.commands[:, :3].abs() < self.xy_velocity_threshold).all(dim=-1))
        self.feet_air_time *= (~(self.feet_contact_state > 0.5)).to(torch.int)

        # cosmetic penalty for hip motion
        rew_hip = torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1) * \
                  self.rew_scales["hip"]

        rew_dof_bias = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1) * self.rew_scales["dof_bias"]

        # survival reward, ensures that the reward is always positive
        # rew_survival = self.progress_buf / (self.max_episode_length - 1)
        # rew_survival = 20.0 * self.commands[0, 0]
        rew_survival = 0.0 * self.dt

        buf_length = 2
        body_height_buf = self.obs_buffer_dict["bodyPos"].get_len_data_raw(buf_length)[:, 2, :]
        lin_vel_buf = self.obs_buffer_dict["linearVelocity"].get_len_data_raw(buf_length)
        ang_vel_buf = self.obs_buffer_dict["angularVelocity"].get_len_data_raw(buf_length)
        projectedGravity_buf = self.obs_buffer_dict["projectedGravity"].get_len_data_raw(buf_length)
        motor_vel_buf = self.obs_buffer_dict["dofVelocity"].get_len_data_raw(buf_length)
        motor_torque_buf = self.obs_buffer_dict["motorTorque"].get_len_data_raw(buf_length)
        motor_torque_buf2 = self.obs_buffer_dict["motorTorque"].get_len_data_raw(buf_length)
        feet_force_buf = self.obs_buffer_dict["feetForce"].get_len_data_raw(buf_length)
        roll_ang_buf = self.obs_buffer_dict["rollAngle"].get_len_data_raw(buf_length).squeeze()
        pitch_ang_buf = self.obs_buffer_dict["pitchAngle"].get_len_data_raw(buf_length).squeeze()
        v_x_buf = lin_vel_buf[:, 0, :]
        v_y_buf = lin_vel_buf[:, 1, :]
        v_z_buf = lin_vel_buf[:, 2, :]
        v_roll_buf = ang_vel_buf[:, 0, :]
        v_pitch_buf = ang_vel_buf[:, 1, :]
        v_yaw_buf = ang_vel_buf[:, 2, :]
        roll_sin_buf = projectedGravity_buf[:, 0, :]
        pitch_sin_buf = projectedGravity_buf[:, 1, :]
        power_mech_buf = motor_torque_buf * motor_vel_buf
        power_heat_buf = 0.26 * motor_torque_buf * motor_torque_buf
        power = power_mech_buf + power_heat_buf
        # power = power_mech_buf
        power = torch.clip(power, min=0., max=None)
        feet_resultant_force_buf = torch.norm(feet_force_buf.view(self.num_envs, 4, 3, -1), dim=2)
        # feet_resultant_force_buf = feet_force_buf.view(self.num_envs, 4, 3, -1)[:, :, 2, :]
        feet_max_force_each = torch.max(feet_resultant_force_buf, dim=2).values
        feet_max_force_std = torch.std(feet_max_force_each, dim=-1)
        feet_max_force_total = torch.max(feet_max_force_each, dim=-1).values
        body_height_mean = torch.mean(body_height_buf, dim=-1)
        v_x_mean = torch.mean(v_x_buf, dim=-1)
        v_y_mean = torch.mean(v_y_buf, dim=-1)
        v_z_mean = torch.mean(v_z_buf, dim=-1)
        v_roll_mean = torch.mean(v_roll_buf, dim=-1)
        v_pitch_mean = torch.mean(v_pitch_buf, dim=-1)
        v_yaw_mean = torch.mean(v_yaw_buf, dim=-1)
        v_x_std = torch.std(v_x_buf, dim=-1)
        v_y_std = torch.std(v_y_buf, dim=-1)
        v_z_std = torch.std(v_z_buf, dim=-1)
        v_roll_std = torch.std(v_roll_buf, dim=-1)
        v_pitch_std = torch.std(v_pitch_buf, dim=-1)
        v_yaw_std = torch.std(v_yaw_buf, dim=-1)
        roll_sin_mean = torch.mean(roll_sin_buf, dim=-1)
        pitch_sin_mean = torch.mean(pitch_sin_buf, dim=-1)
        roll_ang_mean = torch.mean(roll_ang_buf, dim=-1)
        pitch_ang_mean = torch.mean(pitch_ang_buf, dim=-1)
        roll_ang_std = torch.std(roll_ang_buf, dim=-1)
        pitch_ang_std = torch.std(pitch_ang_buf, dim=-1)
        power_mean_each = torch.mean(power, dim=-1)
        power_mean_total = torch.sum(power_mean_each, dim=-1)
        power_max_mean_each = torch.max(power_mean_each, dim=-1).values
        power_max_mean_std = torch.std(power_mean_each[:, [1, 2, 4, 5, 7, 8, 10, 11]], dim=-1)
        torque_max_each = torch.max(torch.abs(motor_torque_buf2), dim=-1).values
        torque_max_each_mean = torch.mean(torque_max_each, dim=-1)
        torque_max_each_std = torch.std(torque_max_each[:, [1, 2, 4, 5, 7, 8, 10, 11]], dim=-1)

        tmp_vel_mean = torch.stack([v_x_mean, v_y_mean, v_z_mean, v_roll_mean, v_pitch_mean, v_yaw_mean], dim=1)
        # tmp_std = torch.stack([v_x_std, v_y_std, v_z_std, v_roll_std, v_pitch_std, v_yaw_std, roll_ang_std, pitch_ang_std], dim=1)
        self.vel_average[:] = tmp_vel_mean * 1. + self.vel_average * 0.
        # print(f"vx_average: {self.vel_average[:, 0]}")
        self.vx_mean[:] = v_x_mean.unsqueeze(-1)


        # print(f"v_x_mean: {v_x_mean}")
        # print(f"v_y_mean: {v_y_mean}")
        # print(f"v_yaw_mean: {v_yaw_mean}")
        # print(self.obs_buffer_dict["linearVelocity"].get_latest_data_raw()[0][0])

        # energy efficiency penalty
        energy_cot = torch.where(v_x_mean != 0, power_mean_total / (12.776 * 9.8 * torch.abs(v_x_mean)), power_mean_total / 40.0)
        rew_energy = torch.exp(-torch.square(energy_cot) * 0.25) * self.rew_scales["energy"]
        self.power_norm[:] = energy_cot.unsqueeze(-1)
        # print(f"rew_energy: {rew_energy[0]}")

        # rew_base_height = torch.square(body_height_mean - self.desired_base_height) * self.rew_scales["base_height"]

        # rew std
        # rew_fallen_over = torch.sum(tmp_std, dim=-1) * self.rew_scales["fallen_over"]
        # rew_fallen_over = torch.sum(torch.square(self.actions), dim=-1) * self.rew_scales["fallen_over"]
        # rew_lin_vel_z = torch.square(self.vel_average[:, 2]) * self.rew_scales["lin_vel_z"]
        # rew_ang_vel_xy = torch.sum(torch.square(self.vel_average[:, 3:5]), dim=1) * self.rew_scales["ang_vel_xy"]
        # rew_orient = (torch.square(roll_ang_mean) + torch.square(pitch_ang_mean)) * self.rew_scales["orient"]

        lin_vel_error = torch.square(self.commands[:, 0] - self.vel_average[:, 0]) + torch.square(self.commands[:, 1] - self.vel_average[:, 1])
        # rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        # rew_lin_vel_xy = -lin_vel_error * self.rew_scales["lin_vel_xy"]
        ang_vel_error = torch.square(self.commands[:, 2] - self.vel_average[:, 5])
        # rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]
        # rew_ang_vel_z = -ang_vel_error * self.rew_scales["ang_vel_z"]
        rew_power = power_mean_total * self.rew_scales["power"]
        # power_mean_total = torch.clip(power_mean_total, min=0., max=900.)
        # rew_power = torch.where(torch.abs(tmp_lin_vel_error[:, 0]) < 0.1, 900.0 - power_mean_total, torch.zeros_like(power_mean_total)) * self.rew_scales["power"]
        # print(power_mean_total)

        rew_power_max = power_max_mean_each * self.rew_scales["power_max_mean_each"]
        rew_power_max_std = power_max_mean_std * self.rew_scales["power_max_mean_std"]
        rew_feet_force_max = feet_max_force_total * self.rew_scales["feet_max_force_total"]
        rew_feet_force_max_std = feet_max_force_std * self.rew_scales["feet_max_force_std"]
        rew_torque_max = torque_max_each_mean * self.rew_scales["torque_max_mean_each"]
        rew_torque_max_std = torque_max_each_std * self.rew_scales["torque_max_mean_std"]

        # rew_lin_vel_xy = -20 * torch.abs(self.vel_average[:, 0] - self.commands[:, 0]) - torch.square(self.vel_average[:, 1] - self.commands[:, 1])
        # rew_ang_vel_z = -1.0 * torch.square(self.vel_average[:, 5] - self.commands[:, 2])
        # rew_energy = -0.04 * energy

        # imitation
        rew_imitation_torque = 0.
        rew_imitation_joint_pos = 0.
        rew_imitation_joint_vel = 0.
        # imitation_torque_error = torch.sum(torch.square(self.ref_tau_ff - self.action_tau_ff), dim=1)
        # # imitation_joint_pos_error = torch.sum(torch.square(self.ref_dof_pos - self.action_dof_pos) + torch.square(self.ref_dof_pos - self.dof_pos), dim=1)
        imitation_joint_pos_error = torch.sum(torch.square(self.ref_dof_pos - self.action_dof_pos), dim=1)
        # imitation_joint_vel_error = torch.sum(torch.square(self.ref_dof_vel - self.dof_vel), dim=1)
        # rew_imitation_torque = imitation_torque_error / 2500. * self.rew_scales["imitation_torque"]
        rew_imitation_joint_pos = imitation_joint_pos_error / 5 * self.rew_scales["imitation_joint_pos"]
        # rew_imitation_joint_vel = torch.exp(-imitation_joint_vel_error / 50.) * self.rew_scales["imitation_joint_vel"]
        #
        # rew_imitation_joint_pos = torch.exp(-torch.sum(torch.square(self.ref_dof_pos - self.dof_pos), dim=-1)) * self.rew_scales["imitation_joint_pos"]
        # rew_imitation_joint_vel = torch.exp(-torch.sum(torch.square((self.ref_dof_vel - self.dof_vel) * 0.05), dim=-1)) * self.rew_scales["imitation_joint_vel"]
        # imitation_torque_error = torch.sum(torch.abs(self.ref_tau_ff - self.Kp * (self.action_dof_pos - self.ref_dof_pos)) / (torch.abs(self.ref_tau_ff) + 1.e-3), dim=-1)
        # rew_imitation_torque = torch.exp(-imitation_torque_error / 20.) * self.rew_scales["imitation_torque"]

        # feet contact regulating (use C des)
        # feet_force_norm = torch.norm(self.feet_force.view(self.num_envs, 4, 3), p=2, dim=-1)
        # feet_vel_xy_world_norm = torch.norm(self.feet_lin_vel_world.view(self.num_envs, 4, 3)[..., :2], p=2, dim=-1)
        # f_coef = (1. - self.ref_phase_C_des) * (1 - torch.exp(-feet_force_norm / 50.))
        # vxy_coef = self.ref_phase_C_des * (1 - torch.exp(-feet_vel_xy_world_norm / 1.))
        # rew_feet_contact_regulate = torch.sum(f_coef + vxy_coef, dim=1) * self.rew_scales["feet_contact_regulate"]

        env_id_leg_broken_count_0 = torch.where(self.leg_broken_count == 0)[0]
        env_id_leg_broken_count_not_0 = torch.where(self.leg_broken_count > 0)[0]
        env_id_leg_broken_count_1 = torch.where(self.leg_broken_count == 1)[0]
        env_id_leg_broken_count_2 = torch.where(self.leg_broken_count == 2)[0]

        feet_force_norm = torch.norm(self.feet_force.view(self.num_envs, 4, 3), p=2, dim=-1)
        # feet_force_norm = torch.where(feet_force_norm > 0.001, 0.02, feet_force_norm)
        err_swing_schedule = torch.square(feet_force_norm)
        # err_swing_schedule = torch.where((0.001 < err_swing_schedule) & (err_swing_schedule < 1.0), 1.0, err_swing_schedule)
        rew_swing_schedule = torch.sum((1. - self.ref_phase_C_des_leg_broken) * (torch.exp(-err_swing_schedule * 0.02) - 1.0), dim=1) / 4.0
        # stance schedule reward
        feet_vel_xy_world_norm = torch.norm(self.feet_lin_vel_world.view(self.num_envs, 4, 3)[..., :2], p=2, dim=-1)
        err_stance_schedule = torch.square(feet_vel_xy_world_norm)
        rew_stance_schedule = torch.sum(self.ref_phase_C_des_leg_broken * (torch.exp(-err_stance_schedule * 0.8) - 1.0), dim=1) / 4.0
        # contact schedule reward
        rew_feet_contact_regulate = (rew_swing_schedule + rew_stance_schedule) * self.rew_scales["feet_contact_regulate"]



        # foothold
        num_feet = len(self.feet_indices)
        lift_height_error = (torch.abs(self.ref_phase_norm_current-0.75) < 0.015) * (self.feet_position_world.view(self.num_envs, num_feet, 3)[..., 2] - 0.07)
        lift_height_error = 0.
        # tmp_vel = self.base_lin_vel.repeat(1, num_feet).view(self.num_envs, num_feet, 3)
        # tmp_vel[..., 1] += self.base_ang_vel[:, 2].unsqueeze(-1) * 0.1805 * self.side_coef
        # tmp_vel_horizon_frame = torch.zeros_like(tmp_vel)
        # for i in range(num_feet):
        #     tmp_vel_horizon_frame[:, i, :] = quat_rotate(self.base_quat_horizon, tmp_vel[:, i, :])
        # ref_foothold_xy = tmp_vel_horizon_frame[..., :2] * (self.gait_periods * self.gait_commands[:, 1]).unsqueeze(-1).unsqueeze(-1) * 0.5
        # tmp_vel_horizon_frame = self.calculate_vel_horizon_frame(self.base_lin_vel, self.base_ang_vel, self.base_lin_vel_command, self.base_ang_vel_command, self.base_quat_horizon, vel_weight=0.8)
        # self.calculate_ref_foot_xy(self.ref_phase_norm_current, tmp_vel_horizon_frame[..., :2], self.gait_periods, self.gait_duty)
        # print(f"vxy: {tmp_vel_horizon_frame[0, 0]}")
        # print(f"ref_foot_pos: {self.ref_foot_pos_xy_body[0, 0, 0]}")
        # print(f"foot_pos_track_weight: {self.foot_pos_track_weight[0, 0]}")
        foothold_error = self.foot_pos_track_weight * torch.norm(self.feet_position_moved_hip_horizon_frame.view(self.num_envs, 4, 3)[..., :2] - self.ref_foot_pos_xy_horizon, p=2, dim=-1)
        rew_imitation_foot_pos = (self.gait_commands_count > 1) * torch.sum(torch.square((foothold_error + lift_height_error) * 100.), dim=-1) * self.rew_scales["imitation_foot_pos"]
        foot_vel_error = self.foot_pos_track_weight * torch.norm(self.feet_velocity_hip_horizon_frame.view(self.num_envs, 4, 3)[..., :2] - self.ref_foot_vel_xy_horizon, p=2, dim=-1)
        rew_imitation_foot_vel = (self.gait_commands_count > 1) * torch.sum(torch.square(foot_vel_error * 6.), dim=-1) * self.rew_scales["imitation_foot_vel"]

        # print(f"ref_phase_norm_current: {self.ref_phase_norm_current[0]}")
        # print(f"lift_height_error: {lift_height_error[0]}")
        # print(f"foothold_error: {foothold_error[0]}")
        # print(f"ref_foothold: {ref_foothold[0, :, 0]}")

        # value_lin_vel_xy = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # value_ang_vel_z = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # value_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        # rew_lin_vel_xy = torch.exp(-8. * value_lin_vel_xy) * 0.15
        # rew_ang_vel_z = torch.exp(-8. * value_ang_vel_z) * 0.15
        # rew_ang_vel_xy = torch.exp(-8. * value_ang_vel_xy) * 0.15
        # rew_cmd = torch.exp(-8. * (value_lin_vel_xy + value_ang_vel_z + value_ang_vel_xy)) * 0.1
        # rew_torques = torch.exp(-torch.sum(torch.square(self.torques / 20.), dim=1)) * 0.1
        # rew_delta_torques = torch.exp(-torch.sum(torch.square((self.torques - self.last_torques) / 20.), dim=1)) * 0.1
        # rew_orient = torch.exp(-torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)) * 0.05
        # rew_base_height = torch.exp(-80. * torch.square(self.root_states[:, 2] - self.height_commands.squeeze())) * 0.05
        # rew_imitation_joint_pos = torch.exp(-2.0 * torch.sum(torch.square(self.ref_dof_pos - self.dof_pos), dim=1)) * 0.125
        # rew_imitation_joint_vel = torch.exp(-2.0 * self.dt * torch.sum(torch.square(self.ref_dof_vel - self.dof_vel), dim=1)) * 0.375
        # # value_feet_contact_regulate = torch.sum((1. - self.ref_phase_C_des) * feet_force_norm * 0.0064 + self.ref_phase_C_des * feet_vel_xy_world_norm * 4.0, dim=-1)
        # rew_feet_contact_regulate = torch.exp(-torch.sum(f_coef + vxy_coef, dim=1) * 0.6) * 0.05

        # terrain type
        # if self.custom_origins:
        #     env_ids_not_plane = (self.terrain_types > 1).nonzero(as_tuple=False).flatten()
        #     env_ids_plane = (self.terrain_types < 2).nonzero(as_tuple=False).flatten()
        #     rew_base_height[env_ids_not_plane] *= 0.
        #     rew_ang_vel_z[env_ids_not_plane] /= 2.0
        #     rew_ang_vel_xy[env_ids_not_plane] /= 4.0
        #     rew_orient[env_ids_not_plane] *= 0.0
        #     # rew_torques[env_ids_not_plane] /= 2.0
        #     rew_imitation_foot_pos[env_ids_not_plane] /= 10.0
        #     rew_imitation_foot_vel[env_ids_not_plane] /= 10.0
        #
        #     rew_stumble[env_ids_plane] = 0.
        #     rew_action_rate[env_ids_plane] = 0.
        # else:
        #     rew_stumble[:] = 0.
        #     rew_action_rate[:] = 0.

        # print(torch.sum((feet_force_norm * feet_vel_xy_world_norm), dim=1)[0])
        # rew_feet_contact_regulate[env_id_leg_broken_count_1] = -torch.sum((feet_force_norm[env_id_leg_broken_count_1] * feet_vel_xy_world_norm[env_id_leg_broken_count_1]), dim=1) * 0.01 * 0.02
        rew_feet_contact_regulate[env_id_leg_broken_count_not_0] = 0.0
        # rew_feet_contact_regulate[env_id_leg_broken_count_2] *= 0.01
        # rew_feet_contact_regulate += -torch.sum((feet_force_norm * self.leg_broken_flag > 0.01).to(torch.float), dim=1) * 0.2

        self.feet_height_rel_ground[:] = self.feet_position_world.view(self.num_envs, 4, 3)[:, :, 2] - self.get_heights_xy(self.feet_position_world.view(self.num_envs, 4, 3)[:, :, :2])
        err_feet_height_leg_broken = torch.clip((self.feet_height_rel_ground - 0.05) * self.leg_broken_flag, min=None, max=0.0)
        # rew_feet_height_leg_broken = (torch.exp(-torch.sum(torch.square(err_feet_height_leg_broken) * 1000., dim=-1)) - 1.0) * 0.02
        rew_feet_height_leg_broken = -torch.sum(torch.square(err_feet_height_leg_broken), dim=-1) * 600.0 * 0.02
        # rew_feet_height_leg_broken = 0.0

        rew_hip[env_id_leg_broken_count_not_0] = 0.0
        # rew_hip[env_id_leg_broken_count_2] = 0.0
        rew_dof_bias[env_id_leg_broken_count_not_0] = 0.0
        # rew_feet_contact_regulate[env_id_leg_broken_count_0] = 0.0
        rew_lin_vel_xy[env_id_leg_broken_count_2] = torch.exp(-torch.sum(torch.square(
            self.commands[env_id_leg_broken_count_2, :2] - self.horizon_lin_vel[env_id_leg_broken_count_2, :2]),
                                                                         dim=1) / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z[env_id_leg_broken_count_2] = torch.exp(-torch.square(
            self.commands[env_id_leg_broken_count_2, 2] - self.horizon_ang_vel[env_id_leg_broken_count_2, 2]) / 0.25) * \
                                                   self.rew_scales["ang_vel_z"]
        rew_lin_vel_z[env_id_leg_broken_count_2] = torch.square(self.horizon_lin_vel[env_id_leg_broken_count_2, 2]) * \
                                                   self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy[env_id_leg_broken_count_2] = torch.sum(
            torch.square(self.horizon_ang_vel[env_id_leg_broken_count_2, :2]), dim=1) * self.rew_scales["ang_vel_xy"]
        rew_orient[env_id_leg_broken_count_2] = 0.0
        rew_delta_torques[env_id_leg_broken_count_0] = 0.0
        rew_joint_acc[env_id_leg_broken_count_0] = 0.0

        rew_base_height = torch.square(10.0 * self.body_traj_error[:, 2]) * self.rew_scales["base_height"]
        rew_base_height[env_id_leg_broken_count_2] = 0.0
        rew_imitation_foot_pos[env_id_leg_broken_count_not_0] = 0.0
        rew_imitation_foot_vel[env_id_leg_broken_count_not_0] = 0.0

        motor_fault_flag = (~self.motor_not_broken_flag.bool()).to(torch.long)
        rew_fault_joint_vel = (-torch.sum(torch.square(self.dof_vel*motor_fault_flag*0.25), dim=1)) * 0.02
        rew_fault_joint_vel = 0

        rew_survival = 0.0 * self.dt
        # total reward
        self.rew_buf[:] = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height + \
                          rew_torques + rew_joint_acc + rew_knee_collision + rew_action_rate + rew_air_time + rew_hip + rew_stumble + rew_energy + rew_power + rew_survival + \
                          rew_power_max + rew_power_max_std + rew_feet_force_max + rew_feet_force_max_std + rew_torque_max + rew_torque_max_std + rew_fallen_over + rew_delta_torques + rew_dof_bias + \
                          rew_gait_tracking + rew_gait_trans_rate + rew_gait_phase_timing + rew_gait_phase_shape + \
                          rew_imitation_torque + rew_imitation_joint_pos + rew_imitation_joint_vel + rew_imitation_foot_pos + rew_imitation_foot_vel + rew_feet_contact_regulate + rew_feet_height_leg_broken + rew_fault_joint_vel
        # self.rew_buf[:] = rew_energy
        # self.rew_buf[:] = rew_lin_vel_xy + rew_ang_vel_z + rew_ang_vel_xy + rew_torques + rew_delta_torques + rew_orient + rew_base_height + rew_imitation_joint_pos + rew_imitation_joint_vel + rew_feet_contact_regulate
        # print(self.rew_buf)
        # print(f"torques: {rew_torques}")
        # print(f"energy: {rew_energy}")
        # print(f"euler: {self.euler_xyz}")

        self.rew_buf[:] = torch.clip(self.rew_buf, min=0., max=None)

        # add termination reward
        # self.rew_buf[:] += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        # log episode reward sums
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["torques"] += rew_torques
        self.episode_sums["delta_torques"] += rew_delta_torques
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["knee_collision"] += rew_knee_collision
        self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["air_time"] += rew_air_time
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["hip"] += rew_hip
        self.episode_sums["dof_bias"] += rew_dof_bias
        self.episode_sums["energy"] += rew_energy
        self.episode_sums["power"] += rew_power
        self.episode_sums["power_max_mean_each"] += rew_power_max
        self.episode_sums["power_max_mean_std"] += rew_power_max_std
        self.episode_sums["feet_max_force_total"] += rew_feet_force_max
        self.episode_sums["feet_max_force_std"] += rew_feet_force_max_std
        self.episode_sums["torque_max_mean_each"] += rew_torque_max
        self.episode_sums["torque_max_mean_std"] += rew_torque_max_std
        self.episode_sums["fallen_over"] += rew_fallen_over
        self.episode_sums["gait_tracking"] += rew_gait_tracking
        self.episode_sums["gait_trans_rate"] += rew_gait_trans_rate
        self.episode_sums["gait_phase_timing"] += rew_gait_phase_timing
        self.episode_sums["gait_phase_shape"] += rew_gait_phase_shape
        self.episode_sums["imitation_torque"] += rew_imitation_torque
        self.episode_sums["imitation_joint_pos"] += rew_imitation_joint_pos
        self.episode_sums["imitation_joint_vel"] += rew_imitation_joint_vel
        self.episode_sums["imitation_foot_pos"] += rew_imitation_foot_pos
        self.episode_sums["imitation_foot_vel"] += rew_imitation_foot_vel
        self.episode_sums["feet_contact_regulate"] += rew_feet_contact_regulate
        self.episode_sums["feet_height_leg_broken"] += rew_feet_height_leg_broken
        self.episode_sums["fault_joint_vel"] += rew_fault_joint_vel

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # positions_offset = 1.0
        velocities = 0.0

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.dof_vel[env_ids] = velocities

        # self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        # self.dof_vel[env_ids] = 0.

        self.dof_pos_rel_init[env_ids] = self.dof_pos[env_ids] - self.default_dof_pos[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
        self.ground_height_current[env_ids] = 0.0
        self.init_position_bias_rel_world[env_ids, :2] = self.root_states[env_ids, :2]
        self.init_position_bias_rel_world[env_ids, 2] = self.get_heights_xy(self.root_states[env_ids, :2])

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1],
        #                                              (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1],
        #                                              (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1],
        #                                              (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 2] = self.commands[env_ids, 3]
        # self.commands[env_ids, 1] = 0.
        # self.commands[env_ids, 2] = 0.
        # self.commands[env_ids, 0] = 1.0
        # self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.xy_velocity_threshold).unsqueeze(1)  # set small commands to zero. wsh_annotation: TODO 0.25 ?
        # self.modify_vel_command()
        # self.commands[env_ids] *= ~((self.commands[env_ids, :3].abs() < self.xy_velocity_threshold).all(dim=-1)).unsqueeze(1)
        self.commands_last[env_ids] = 0.
        # self.modify_vel_command()

        # self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = self.dof_vel[env_ids]
        self.feet_air_time[env_ids] = 0.
        # self.progress_buf[env_ids] = 0
        # self.reset_buf[env_ids] = 1  ### wsh_annotation: TODO

        ### wsh_annotation: TODO(completed) reset to acquire the initial obs_buf
        # self.gym.simulate(self.sim)
        # if self.device == 'cpu':
        #     self.gym.fetch_results(self.sim, True)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.euler_xyz[env_ids] = get_euler_xyz2(self.base_quat[env_ids])
        tmp_q = self.dof_pos.clone().reshape(self.num_envs, 4, 3)
        self.kinematic_feet_pos[:], self.jacobian[:], self.inverse_jacobian[:] = self.robot_kinematics.forward_kinematics(tmp_q)

        # self.update_horizon_quat(env_ids)
        self.base_quat_horizon[env_ids] = quat_from_euler_xyz(self.euler_xyz[env_ids, 0], self.euler_xyz[env_ids, 1], torch.zeros_like(self.euler_xyz[env_ids, 2]))
        self.horizon_quat_in_world[env_ids] = quat_from_euler_xyz(torch.zeros_like(self.euler_xyz[env_ids, 0]),
                                                                  torch.zeros_like(self.euler_xyz[env_ids, 1]),
                                                                  self.euler_xyz[env_ids, 2])
        self.horizon_lin_vel[env_ids] = quat_rotate_inverse(self.horizon_quat_in_world[env_ids], self.root_states[env_ids, 7:10])
        self.horizon_ang_vel[env_ids] = quat_rotate_inverse(self.horizon_quat_in_world[env_ids], self.root_states[env_ids, 10:13])
        # base_quat_horizon2 = self.base_quat_horizon.clone()
        # world2base_quat = self.base_quat.clone()
        # world2base_quat[:, :3] = -world2base_quat[:, :3].clone()
        # base_quat_horizon2[env_ids] = quat_mul(world2base_quat[env_ids], quat_from_euler_xyz(torch.zeros_like(self.euler_xyz[env_ids, 0]), torch.zeros_like(self.euler_xyz[env_ids, 1]), self.euler_xyz[env_ids, 2]))

        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        self.ref_body_trajectory[env_ids] = 0.
        self.ref_body_trajectory[env_ids, 2] = self.height_commands[env_ids, 0] + self.ground_height_current[env_ids]
        self.act_body_trajectory[env_ids, :3] = self.root_states[env_ids, :3] - self.init_position_bias_rel_world[env_ids]
        self.act_body_trajectory[env_ids, 3:6] = self.euler_xyz[env_ids]
        self.act_body_trajectory[env_ids, 6:8] = self.root_states[env_ids, 7:9]
        self.act_body_trajectory[env_ids, 8] = self.base_lin_vel[env_ids, 2]
        self.act_body_trajectory[env_ids, 9:] = self.base_ang_vel[env_ids]
        self.body_traj_error[env_ids] = self.ref_body_trajectory[env_ids] - self.act_body_trajectory[env_ids]

        # b = torch.tensor([0, 1], dtype=torch.long, device=self.device, requires_grad=False)
        # a = self.rigid_body_states_reshape[b][:, self.feet_indices, :]
        # aa = a[..., 0:3]
        # aaa = aa.view(len(env_ids), -1)
        self.feet_position_world[env_ids] = self.rigid_body_states_reshape[env_ids][:, self.feet_indices, 0:3].reshape(len(env_ids), 12)
        self.feet_lin_vel_world[env_ids] = self.rigid_body_states_reshape[env_ids][:, self.feet_indices, 7:10].reshape(len(env_ids), 12)
        for i in range(len(self.feet_indices)):
            self.feet_position_body_in_world_frame[env_ids, i * 3: i * 3 + 3] = self.feet_position_world[env_ids, i * 3: i * 3 + 3] - self.root_states[env_ids, 0:3]
            self.feet_position_body[env_ids, i * 3: i * 3 + 3] = quat_rotate_inverse(self.base_quat[env_ids], self.feet_position_world[env_ids, i * 3: i * 3 + 3] - self.root_states[env_ids, 0:3])
            self.feet_lin_vel_body[env_ids, i * 3: i * 3 + 3] = quat_rotate_inverse(self.base_quat[env_ids],self.feet_lin_vel_world[env_ids, i * 3: i * 3 + 3] - self.root_states[env_ids, 7:10])
        self.feet_position_hip[env_ids] = self.feet_position_body[env_ids] - self.hip_position_rel_body


        # self.feet_position_hip[env_ids] = self.kinematic_feet_pos[env_ids].reshape(len(env_ids), 12)
        # tmp_dq = self.dof_vel.clone().reshape(self.num_envs, 4, 3)
        # self.feet_lin_vel_body[env_ids] = torch.matmul(self.jacobian[env_ids], tmp_dq[env_ids].unsqueeze(-1)).squeeze(-1).reshape(len(env_ids), 12)
        # self.feet_position_body[env_ids] = self.feet_position_hip[env_ids] + self.hip_position_rel_body

        self.feet_position_hip_from_joint[env_ids] = self.kinematic_feet_pos[env_ids].reshape(len(env_ids), 12)
        tmp_dq = self.dof_vel.clone().reshape(self.num_envs, 4, 3)
        self.feet_lin_vel_body_from_joint[env_ids] = torch.matmul(self.jacobian[env_ids], tmp_dq[env_ids].unsqueeze(-1)).squeeze(-1).reshape(len(env_ids), 12)

        self.feet_position_moved_hip[env_ids] = self.feet_position_hip[env_ids] - self.leg_bias_rel_hip

        for i in range(len(self.feet_indices)):
            self.feet_position_hip_horizon_frame[env_ids, i * 3: i * 3 + 3] = quat_rotate(self.base_quat_horizon[env_ids], self.feet_position_hip[env_ids, i * 3: i * 3 + 3])
            self.feet_velocity_hip_horizon_frame[env_ids, i * 3: i * 3 + 3] = quat_rotate(
                self.base_quat_horizon[env_ids], self.feet_lin_vel_body[env_ids, i * 3: i * 3 + 3])
            self.feet_position_moved_hip_horizon_frame[env_ids, i * 3: i * 3 + 3] = quat_rotate(self.base_quat_horizon[env_ids], self.feet_position_moved_hip[env_ids, i * 3: i * 3 + 3])

        self.force_ff_mpc[env_ids] = 0.0
        self.est_feet_force[env_ids] = 0.0
        self.ref_feet_force[env_ids] = 0.0
        self.feet_lin_momentum[env_ids] = 0.0
        self.ref_feet_lin_momentum[env_ids] = 0.0
        self.feet_ang_momentum[env_ids] = 0.0
        self.ref_feet_ang_momentum[env_ids] = 0.0

        # calculate the linear acceleration of the base
        self.base_lin_acc[env_ids] = 0.
        self.base_lin_acc[env_ids] -= self.gravity_acc
        self.last_base_lin_vel_rel_world[env_ids] = self.root_states[env_ids, 7:10].clone().detach()

        self.actions[env_ids] = 0.0
        self.last_actions_raw[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_torques[env_ids] = 0.
        self.feet_contact_state[env_ids] = 0
        self.feet_contact_state_obs[env_ids] = -0.5
        self.feet_force[env_ids] = self.contact_forces[env_ids][:, self.feet_indices].reshape(len(env_ids), 12)
        self.feet_force[env_ids] = 0.0

        self.ref_phase_contact_state[env_ids] = 0.0
        self.ref_phase_contact_num[env_ids] = 0.0
        self.ref_feet_force[env_ids] = 0.0

        self.controller_reset_buf[env_ids] = 1

        self.height_commands[env_ids] = 0.3

        self.motor_not_broken_flag[env_ids] = 1
        self.leg_broken_flag[env_ids] = 0
        self.leg_not_broken_flag[env_ids] = (~self.leg_broken_flag[env_ids].bool()).to(torch.long)
        self.motor_broken_count1[env_ids] = torch.randint(0, 1, (len(env_ids),), dtype=torch.long, device=self.device)
        self.motor_broken_count2[env_ids] = torch.randint(0, 1, (len(env_ids),), dtype=torch.long, device=self.device)
        self.leg_broken_count[env_ids] = torch.sum(self.leg_broken_flag[env_ids], dim=1)
        self.gait_index_leg_broken[env_ids] = self.index_leg_broken_flag_to_gait[self.leg_broken_flag[env_ids].transpose(-2, -1).unbind(0)]
        self.gait_params_leg_broken[env_ids] = self.gait_tensor_leg_broken[self.gait_index_leg_broken[env_ids]]

        # self.gait_period_offset[env_ids] = (torch.zeros_like(self.feet_contact_state) + torch_rand_float(-0.23, 0.1, (self.num_envs, 1), device=self.device))[env_ids]
        # self.gait_period_offset[env_ids] = -0.2
        # self.gait_duty_cycle_offset[env_ids] = (torch.zeros_like(self.feet_contact_state) - 0.0)[env_ids]
        # self.gait_phase_offset[env_ids] = (torch.zeros_like(self.feet_contact_state))[env_ids]
        #
        # self.motion_planning_interface.update_gait_planning(True, self.gait_period_offset, self.gait_duty_cycle_offset,
        #                                                     self.gait_phase_offset, None)
        # self.motion_planning_interface.update_body_planning(True, None, None, None, None, self.commands[:, :3])
        # self.motion_planning_interface.update_des_feet_pos_rel_hip(self.des_feet_pos_rel_hip)
        # self.motion_planning_interface.generate_motion_command()

        # self.commands[env_ids, 0] = 0.6
        # self.commands[env_ids, 1] = 0.0
        # self.commands[env_ids, 2] = 0.0
        # self.commands[env_ids, 3] = 0.0

        ### wsh_annotation: reset observation buffer
        for key in self.record_items:
            if key == "commands":  ### wsh_annotation: command history is zero
                self.obs_buffer_dict[key].reset_and_fill_index(env_ids, torch.zeros(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False))
            elif key == "gaitCommands":
                self.obs_buffer_dict[key].reset_and_fill_index(env_ids, torch.tensor(len(env_ids)*[[0.5, 0.5, 0.5, 0.5, 0.0, 0.0]], dtype=torch.float, device=self.device, requires_grad=False))
            elif key == "heightCommands":
                self.obs_buffer_dict[key].reset_and_fill_index(env_ids, torch.tensor(len(env_ids) * [[0.3]], dtype=torch.float, device=self.device,requires_grad=False))
            else:
                self.obs_buffer_dict[key].reset_and_fill_index(env_ids, self.obs_name_to_value[key][env_ids])

        self.vel_average[env_ids] = 0.0
        self.power_norm[env_ids] = 0.0
        self.vx_mean[env_ids] = 0.0
        self.command_lin_vel_x[env_ids] = 0.0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids] / self.max_episode_length_s)  # / self.max_episode_length_s  # (self.progress_buf[env_ids] * self.dt + 1.e-5)
            self.episode_sums[key][env_ids] = 0.
            # print(self.extras["episode"]['rew_' + key])
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

        # self.extras["episode"] = {}
        # for key in self.episode_sums2.keys():
        #     self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums2[key][env_ids] / float(self.max_episode_length))
        #     self.episode_sums2[key][env_ids] = 0.
        #     # print(self.extras["episode"]['rew_' + key])
        # self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

        self.extras["rew_error"] = {}
        for key in self.rew_error.keys():
            self.extras["rew_error"]['err_' + key] = torch.mean(self.rew_error[key][env_ids] / (self.progress_buf[env_ids] + 1.e-5))
            self.rew_error[key][env_ids] = 0.

        self.progress_buf[env_ids] = 0

        self.push_random_count[env_ids] = torch.randint(*self.push_interval, (len(env_ids),), device=self.device, dtype=torch.long)
        self.commands_change_random_count[env_ids] = torch.randint(*self.commands_change_interval, (len(env_ids),), device=self.device, dtype=torch.long)
        self.gait_commands_change_random_count[env_ids] = torch.randint(*self.gait_commands_change_interval, (len(env_ids),), device=self.device, dtype=torch.long)
        self.height_commands_change_random_count[env_ids] = torch.randint(*self.height_commands_change_interval, (len(env_ids),), device=self.device, dtype=torch.long)
        self.gait_commands_count[env_ids] = 0

        self.compute_observations()
        self.gait_params_act_last[env_ids] = torch.zeros_like(self.gait_params_act_last[env_ids])
        self.gait_params_act_raw_last[env_ids] = torch.zeros_like(self.gait_params_act_raw_last[env_ids])
        self.phase_overwrite_last[env_ids] = 0.0
        self.gait_period_tracking_error_last[env_ids] = 0.0

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            self.update_target_points(env_ids)
            self.reached_target[env_ids] = 0
            self.check_target_reached()
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        ### wsh_annotation: TODO 卡BUG loop in last and current level ??
        # self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(
        #     self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25)  ### wsh_annotation: TODO 0.25 ?
        # self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] -= 1 * ((self.reached_target[env_ids] < 0.1) & (distance < torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25))
        self.terrain_levels[env_ids] += 1 * self.reached_target[env_ids]
        # self.terrain_levels[env_ids] = 9

        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

        self.update_target_points(env_ids)
        self.reached_target[env_ids] = 0
        self.check_target_reached()

    def check_target_reached(self):
        self.target_pos_rel[:] = self.target_points[:, :2] - self.root_states[:, :2]
        reach_target = torch.norm(self.target_pos_rel, dim=1) < self.target_threshold
        self.reached_target[:] |= reach_target
        reach_target_ids = reach_target.nonzero(as_tuple=False).flatten()
        self.update_target_points(reach_target_ids)

        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec_norm = self.target_pos_rel / (norm + 1e-5)
        self.target_yaw[:] = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
        self.commands[:, 3] = self.target_yaw.clone()

    def update_target_points(self, env_ids):
        num = len(env_ids)
        if num > 0:
            indices = torch.randint(0, 8, (num,))
            self.target_points[env_ids] = self.target_points_list[self.terrain_levels[env_ids], self.terrain_types[env_ids], indices]
            self.target_pos_rel[env_ids] = self.target_points[env_ids, :2] - self.root_states[env_ids, :2]

    def push_robots(self):
        # wsh_annotation: TODO(!!!) How about add external forces ??
        self.root_states[:, 7:9] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device)  # lin vel x/y
        # self.root_states[:, 7:9] = torch.tensor([0., 1.], device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def push_robots_indexed(self):
        # wsh_annotation: TODO(!!!) How about add external forces ??
        env_ids = (self.progress_buf % self.push_random_count == 0).nonzero(as_tuple=False).flatten()
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.root_states[env_ids, 7:9] = torch_rand_float(self.scheduled_push_velocity_range[0], self.scheduled_push_velocity_range[1], (len(env_ids), 2), device=self.device)  # lin vel x/y
        # self.root_states[env_ids, 7:9] = torch.tensor([0., 1.], device=self.device)
        # self.root_states[env_ids, 7:9] = self.current_push_velocity[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # self.current_push_velocity[env_ids] = self.root_states[env_ids, 7:9].clone()
        # if len(env_ids):
        #     if env_ids[0] == 0:
        #         print("push-push-push-push-push-push-push-push-push-push-push")

    def pre_physics_step(self, actions):
        # t = time.time()
        # self.schedule_delta = 0.5
        self.actions[:] = actions.clone().to(self.device)
        # print(f"action_raw_0: {self.actions[0]}")
        self.last_actions_raw[:] = self.actions.clone()
        self.actions[:] = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.actions[:] *= self.action_scale

        with torch.inference_mode():
            self.leg_broken_policy_actions_raw[:] = self.leg_broken_policy(self.states_buf.clone().detach())
        # self.leg_broken_policy_actions[:] = torch.clamp(self.leg_broken_policy_actions_raw, -self.clip_actions, self.clip_actions)
        # self.leg_broken_policy_actions[:] *= self.action_scale

        if self.common_step_counter < 10000:
            beta = 1.0
        else:
            beta = 0.995 ** (self.common_step_counter - 10000)
        beta = 0.0
        self.mixed_actions_raw[:] = beta*self.leg_broken_policy_actions_raw + (1.0-beta)*self.last_actions_raw
        self.mixed_actions[:] = torch.clamp(self.mixed_actions_raw, -self.clip_actions, self.clip_actions)
        self.mixed_actions[:] *= self.action_scale
        # self.actions[:] = self.leg_broken_policy_actions.clone()
        # print(f"err_action_raw: {(self.extras['ref_actions'][0])}")

        # self.actions[:, [0, 3, 6, 9]] *= 0.2  # hip joint kp is larger

        # dof_pos_desired = self.default_dof_pos
        # joint_vel_des = 0.
        self.action_tau_ff[:] = 0.  # self.actions[:, :12].clone() * 27.
        self.action_dof_pos[:] = self.mixed_actions.clone() + self.default_dof_pos
        # self.action_dof_vel[:] = self.actions[:, 24:36].clone() * 20.
        self.action_dof_vel[:] = 0.

        # self.calculate_ref_dof_commands()

        # gait_period_offset = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) - 0.2
        # gait_phase_offset = torch.zeros_like(gait_period_offset)
        # gait_duty_cycle_offset = torch.zeros_like(gait_period_offset)

        # ------------------------------------------ action to gait modulate ------------------------------------------
        # self.gait_period_act = (self.actions[:, 0].unsqueeze(-1) + 1.0) * 0.5  # [0, 1]
        # duty = (self.actions[:, 1].unsqueeze(-1) + 1.0) * 0.5
        # gait_period_offset = (self.gait_period_act - 0.5).repeat(1, 4)
        # gait_duty_cycle_offset = (duty - 0.5).repeat(1, 4)
        # gait_phase_offset = torch.zeros_like(self.actions[:, 2:6])
        # gait_phase_offset[:, :3] = (self.actions[:, 2:5] + 1.0) * 0.5  # [0, 1]
        # gait_phase_offset[:, 3] = self.calculate_phase(self.actions[:, 5], self.actions[:, 6])
        # self.gait_params_act[:] = torch.cat([self.gait_period_act, duty, gait_phase_offset.clone()], dim=-1)
        # ----------------------------------------------------------------------------------------------------

        # ------------------------------------------ action to gait type ------------------------------------------
        # self.gait_commands[:, :2] = self.actions[:, :2] * 0.25 + 0.45  # [0.2, 0.7]
        # self.gait_commands[:, 2:5] = (self.actions[:, 2:5] + 1.0) * 0.5  # [0, 1]
        # self.gait_commands[:, 5] = 0.
        # self.gait_periods[:] = self.gait_commands[:, 0]
        # gait_tracked = self.gait_tracking()
        # self.gait_period_act = gait_tracked[:, 0].unsqueeze(-1)
        # duty = gait_tracked[:, 1].unsqueeze(-1)
        # gait_period_offset = (self.gait_period_act - 0.5).repeat(1, 4)
        # gait_duty_cycle_offset = (duty - 0.5).repeat(1, 4)
        # gait_phase_offset = torch.zeros_like(gait_tracked[:, :4])
        # gait_phase_offset[:, :3] = gait_tracked[:, 2:5].clone()
        # gait_phase_offset[:, 3] = self.calculate_phase(gait_tracked[:, 5], gait_tracked[:, 6])
        # self.gait_params_act[:] = torch.cat([self.gait_period_act, duty, gait_phase_offset.clone()], dim=-1)
        # print(self.gait_params_act[0])

        # ------------------------------------------ action to trajectory ------------------------------------------
        vel_commands = self.commands[:, :3].clone()
        # vel_commands[:, 0] += self.actions[:, 0].clone()
        # # vel_commands[:, 1] = self.actions[:, 1].clone()
        # # vel_commands[:, 2] = self.actions[:, 2].clone()  # * 2.0
        # # self.body_orientation_commands[:, 0] = self.actions[:, 1].clone()  # * 0.2  # [-0.4, 0.4]
        # self.body_orientation_commands[:, 1] = self.actions[:, 1].clone() * 0.6  # * 0.6  # [-0.8, 0.8]
        # self.feet_mid_bias_xy[:, :] = self.actions[:, 2:10].clone() * 0.1  # * 0.1
        # # self.feet_mid_bias_xy[:, 0:2] = self.actions[:, 0].unsqueeze(-1).repeat(1, 2) * 0.1  # [-0.1, 0.1]
        # # self.feet_mid_bias_xy[:, 2:4] = self.actions[:, 1].unsqueeze(-1).repeat(1, 2) * 0.1
        # # self.feet_mid_bias_xy[:, [4, 6]] = self.actions[:, 2].unsqueeze(-1).repeat(1, 2) * 0.1  # [-0.1, 0.1]
        # # self.feet_mid_bias_xy[:, [5, 7]] = self.actions[:, 3].unsqueeze(-1).repeat(1, 2) * 0.1
        # self.feet_lift_height_bias[:, :4] = self.actions[:, 10:14].clone() * 0.2  # * 0.1 - 0.05  # [-0.15, 0.15]
        # self.feet_lift_height_bias[:, 4:8] = self.actions[:, 14:18].clone() * 0.4  # * 0.4  # [-0.4, 0.4]
        # # self.des_feet_pos_rel_hip[:] = self.actions[:, 8:20].clone()  # * 0.15  # [-0.15, 0.15]
        # # self.feet_lift_height_bias[:, :4] = self.actions[:, 0:4].clone() * 0.1  # [-0.15, 0.15]
        # # self.feet_lift_height_bias[:, 4:8] = self.actions[:, 4:8].clone() * 0.4  # [-0.4, 0.4]
        # # self.des_feet_pos_rel_hip[:] = self.actions.clone()  # [-0.15, 0.15]

        # print(self.feet_mid_bias_xy[0])
        # print(self.feet_lift_height_bias[0])
        # print(vel_commands[5, 0])


        # ------------------------------------------- follow gait commands -------------------------------------------
        gait_period_offset = (self.gait_periods - 0.5).unsqueeze(-1)
        gait_duty_cycle_offset = (self.gait_commands[:, 1] - 0.5).unsqueeze(-1)
        gait_phase_offset = self.gait_commands[:, 2:6].clone()
        gait_phase_offset[:, 3] = self.ref_phase_current[:, 1]
        # gait_phase_offset[:, 3] = self.calculate_phase(self.ref_phase_sincos_current[:, 2], self.ref_phase_sincos_current[:, 3])
        # gait_phase_offset[:, 3] = self.calculate_phase(self.actions[:, 0],
        #                                                self.actions[:, 1])
        # ------------------------------------------------------------------------------------------------------------

        # phase0 = self.actions[:, 5].clone()
        # phase1 = self.actions[:, 6].clone()
        # phase2 = self.actions[:, 7].clone()
        # phase3 = self.actions[:, 8].clone()
        # self.gait_commands[:, 2] = phase1 - phase0
        # self.gait_commands[:, 3] = phase2 - phase0
        # self.gait_commands[:, 4] = phase3 - phase0
        # self.gait_commands[:, 5] = phase0.clone()
        # self.gait_commands[:, 2:6] = torch.where(self.gait_commands[:, 2:6] < 0, self.gait_commands[:, 2:6] + 1., self.gait_commands[:, 2:6])
        # gait_period_offset = (self.actions[:, 0] - 0.5).unsqueeze(-1).repeat(1, 4)
        # gait_duty_cycle_offset = self.actions[:, 1:5] - 0.5
        # gait_phase_offset = self.gait_commands[:, 2:6].clone()
        # self.feet_phase_sincos[:, [0, 2, 4, 6]] = torch.sin(self.actions[:, 5:] * torch.pi * 2.)
        # self.feet_phase_sincos[:, [1, 3, 5, 7]] = torch.cos(self.actions[:, 5:] * torch.pi * 2.)
        # print(phase0[0])
        # print(phase1[0])
        # print(phase2[0])
        # print(phase3[0])
        # print("----------------------------------------------------------------")

        # ------------------------------------------- motion planning interface --------------------------------------
        body_height_offset = self.height_commands - 0.3
        gait_phase_offset[:, :2] = gait_phase_offset[:, :2].clone() - 0.5
        # print(self.gait_params_act[0])
        # test motion_planning commands
        # self.body_orientation_commands[:, :] = torch.tensor([0.1, 0.3, 0.003], device=self.device)
        # self.feet_mid_bias_xy[:, :] = torch.tensor([-0., 0.004, 0.007, 0.006, 0.1, -0.1, 0.011, 0.010], device=self.device)
        # self.feet_lift_height_bias[:, :] = torch.tensor([-0.03, 0.02, 0.07, 0.1, 0.017, 0.016, 0.019, 0.018], device=self.device)
        # self.des_feet_pos_rel_hip[:, :] = torch.tensor([0.023, 0.15, 0.025, 0.020, 0.021, 0.022, 0.029, 0.030, 0.031, 0.026, 0.027, 0.028], device=self.device)

        # swing_time = self.actions[:, 0] * 0.1 + 0.2
        # stance_time = self.actions[:, 1] * 0.14 + 0.22
        # self.gait_period_act[:] = swing_time + stance_time
        # duty = stance_time / self.gait_period_act
        # self.gait_period_act = 1.0 / (self.actions[:, 0].unsqueeze(-1) + 3.0)  # [2, 4] Hz
        # duty = self.actions[:, 1].unsqueeze(-1) * 0.2 + 0.5  # [0.3, 0.7]
        # gait_period_offset = (self.gait_period_act.unsqueeze(-1) - 0.5).repeat(1, 4)
        # gait_duty_cycle_offset = (duty.unsqueeze(-1) - 0.5).repeat(1, 4)
        # gait_phase_offset = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        # body_height_offset = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # print(f"period: {self.gait_period_act}")
        # print(f"duty: {duty}")
        # print(f"vx_mean: {self.vx_mean}")

        # self.motion_planning_interface.update_gait_planning(True, gait_period_offset, gait_duty_cycle_offset, gait_phase_offset, None)
        # self.motion_planning_interface.update_body_planning(True, None, self.body_orientation_commands, None, None, vel_commands)
        # # self.motion_planning_interface.update_feet_mid_bias_xy(self.feet_mid_bias_xy)
        # # self.motion_planning_interface.update_feet_lift_height(self.feet_lift_height_bias)
        # # self.motion_planning_interface.update_des_feet_pos_rel_hip(self.des_feet_pos_rel_hip)
        # self.motion_planning_interface.generate_motion_command()
        # # ------------------------------------------------------------------------------------------------------------
        #
        # self.force_ff_mpc[:], torques_est, tau_ff_mpc, q_des, qd_des = self.mit_controller.step_run(self.controller_reset_buf, self.base_quat,
        #                                                                   self.base_ang_vel, self.base_lin_acc,
        #                                                                   self.dof_pos, self.dof_vel,
        #                                                                   self.feet_contact_state,
        #                                                                   self.motion_planning_interface.get_motion_command())  # self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state
        # self.motion_planning_interface.change_gait_planning(False)
        # self.motion_planning_interface.change_body_planning(False)

        # # self.ref_dof_pos[:] = (tau_ff_mpc + self.Kp * q_des + self.Kd * qd_des) / self.Kp
        #
        # tau_ff_in_use = tau_ff_mpc
        # q_des_in_use = q_des
        # qd_des_in_use = qd_des
        tau_ff_in_use = self.action_tau_ff
        q_des_in_use = self.action_dof_pos
        qd_des_in_use = self.action_dof_vel

        # self.ref_feet_force[:, [2, 5, 8, 11]] = self.ref_phase_contact_state * torch.where(
        #     self.ref_phase_contact_num > 0, -self.robot_weight / self.ref_phase_contact_num, 0.0)


        # dof_pos_desired = self.default_dof_pos

        # kp = torch.zeros_like(self.dof_pos[0])
        # kd = torch.zeros_like(self.dof_pos[0])
        # kp[:] = 25
        # kp[[0, 3, 6, 9]] = 5
        # kd[:] = 1
        #
        # v_max = 20.0233
        # v_max /= 1.0
        # tau_max = 33.5 * 1.0
        # k = -3.953886

        self.feet_lin_momentum[:] = 0.
        self.ref_feet_lin_momentum[:] = 0.
        self.feet_ang_momentum[:] = 0.
        self.ref_feet_ang_momentum[:] = 0.

        for i in range(self.decimation - 1):
            # tau_ff_in_use = torch.matmul(self.jacobian.transpose(-2, -1), self.ref_feet_force.reshape(self.num_envs, 4, 3, 1)).squeeze(-1).reshape(self.num_envs, 12)

            torques = self._cal_pd(tau_ff_in_use, q_des_in_use, qd_des_in_use, self.Kp, self.Kd)
            # torques = tau_ff_mpc

            self.est_feet_force[:] = -(self.ref_phase_contact_state.reshape(self.num_envs, 4, 1, 1) * torch.matmul(self.jacobian.transpose(-2, -1).inverse(), torques.reshape(self.num_envs, 4, 3, 1))).squeeze(-1).reshape(self.num_envs, 12)
            self.ref_feet_force_mpc[:] = -(self.ref_phase_contact_state.reshape(self.num_envs, 4, 1) * self.force_ff_mpc.reshape(self.num_envs, 4, 3)).reshape(self.num_envs, 12)

            self.est_feet_force[:] = quat_apply(self.base_quat.repeat(1, 4), self.est_feet_force)
            self.est_feet_force[:, [2, 5, 8, 11]] = self.feet_force[:, [2, 5, 8, 11]]
            self.ref_feet_force_mpc[:] = quat_apply(self.base_quat.repeat(1, 4), self.ref_feet_force_mpc)

            # torques, tau_ff_mpc, q_des, qd_des = self.mit_controller.step_run(self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state, self.motion_planning_interface.get_motion_command())  # self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state
            # self.motion_planning_interface.change_gait_planning(False)
            # self.motion_planning_interface.change_body_planning(False)

            # torques_action = self.action_tau_ff + self.Kp * (self.action_dof_pos - self.dof_pos) + self.Kd * (self.action_dof_vel - self.dof_vel)

            # torques_ref = self.ref_tau_ff + self.Kp * (self.ref_dof_pos - self.dof_pos) + self.Kd * (self.ref_dof_vel - self.dof_vel)
            # torques = torques_action  # torques_ref  # * self.schedule_delta + torques_ref * (1 - self.schedule_delta)
            # torques = torch.clip(torques, -tau_max, tau_max)

            # tau_ff = tau_ff_mpc
            # dof_pos_desired = q_des
            # joint_vel_des = qd_des

            # torques1 = torch.clip(tau_ff + kp * (q_des + dof_pos_comp - self.dof_pos) + kd * (qd_des - self.dof_vel), -tau_max, tau_max)
            # print(f"kp: {kp}")
            # print(f"kd: {kd}")
            # print(f"delta_q_py: {q_des - self.dof_pos}")
            # print(f"torque_estimate: {torques[0]}")
            # print(f"torques1: {torques1[0]}")
            # print(f"tau_ff: {tau_ff[0]}")

            # torques = torch.clip(tau_ff + self.Kp * (dof_pos_desired - self.dof_pos) + self.Kd * (joint_vel_des - self.dof_vel), -tau_max, tau_max)
            # tmp_max_torque = torch.clip(k * (self.dof_vel - v_max), 0, tau_max)
            # tmp_min_torque = torch.clip(k * (self.dof_vel + v_max), -tau_max, 0)
            # torques[:] = torch.where(self.dof_vel > tau_max / k + v_max, torch.clip(torques, -tau_max * torch.ones_like(torques), tmp_max_torque), torques)
            # torques[:] = torch.where(self.dof_vel < -(tau_max / k + v_max), torch.clip(torques, tmp_min_torque, tau_max * torch.ones_like(torques)), torques)
            # print("torques: ", torques)
            # torques = torch.zeros_like(torques)

            torques *= self.motor_not_broken_flag

            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques[:] = torques.view(self.torques.shape)
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.extra_force),
                                                    gymtorch.unwrap_tensor(self.extra_torque), gymapi.ENV_SPACE)
            # self.torques = torques.view(self.torques.shape)
            # if i % 5 == 0 and self.force_render:
            #     self.render()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            # self.gym.refresh_dof_state_tensor(self.sim)
            # self.record_state_test()
            # self.record_state_tensor_test()

            self.update_pre_state()
            # self.record_states_into_buffer()
            self.record_states_duration_sim_dt()
            # self.record_commands()

            # self.torques_square_accumulated += torch.square(torques)

            # self.force_ff_mpc[:], torques_est, tau_ff_mpc, q_des, qd_des = self.mit_controller.step_run(self.controller_reset_buf, self.base_quat,
            #                                                                self.base_ang_vel, self.base_lin_acc,
            #                                                                self.dof_pos, self.dof_vel,
            #                                                                self.feet_contact_state,
            #                                                                self.motion_planning_interface.get_motion_command())
            # self.motion_planning_interface.change_gait_planning(False)
            # self.motion_planning_interface.change_body_planning(False)

        # tau_ff_in_use = torch.matmul(self.jacobian.transpose(-2, -1), self.ref_feet_force.reshape(self.num_envs, 4, 3, 1)).squeeze(-1).reshape(self.num_envs, 12)

        torques = self._cal_pd(tau_ff_in_use, q_des_in_use, qd_des_in_use, self.Kp, self.Kd)
        # torques = tau_ff_mpc

        self.est_feet_force[:] = -(self.ref_phase_contact_state.reshape(self.num_envs, 4, 1, 1) * torch.matmul(self.jacobian.transpose(-2, -1).inverse(), torques.reshape(self.num_envs, 4, 3, 1))).squeeze(-1).reshape(self.num_envs, 12)
        self.ref_feet_force_mpc[:] = -(self.ref_phase_contact_state.reshape(self.num_envs, 4, 1) * self.force_ff_mpc.reshape(self.num_envs, 4, 3)).reshape(self.num_envs, 12)

        self.est_feet_force[:] = quat_apply(self.base_quat.repeat(1, 4), self.est_feet_force)
        self.est_feet_force[:, [2, 5, 8, 11]] = self.feet_force[:, [2, 5, 8, 11]]
        self.ref_feet_force_mpc[:] = quat_apply(self.base_quat.repeat(1, 4), self.ref_feet_force_mpc)
        # print(f"est_feet_force: {self.est_feet_force[0]}")
        # print(f"ref_feet_force_mpc: {self.ref_feet_force_mpc[0]}")

        # torques, tau_ff_mpc, q_des, qd_des = self.mit_controller.step_run(self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state, self.motion_planning_interface.get_motion_command())

        # torques_action = self.action_tau_ff + self.Kp * (self.action_dof_pos - self.dof_pos) + self.Kd * (self.action_dof_vel - self.dof_vel)

        # torques_ref = self.ref_tau_ff + self.Kp * (self.ref_dof_pos - self.dof_pos) + self.Kd * (self.ref_dof_vel - self.dof_vel)
        # torques = torques_action  # torques_ref  # * self.schedule_delta + torques_ref * (1 - self.schedule_delta)
        # torques = torch.clip(torques, -tau_max, tau_max)

        # tau_ff = tau_ff_mpc
        # dof_pos_desired = q_des
        # joint_vel_des = qd_des

        # torques1 = torch.clip(tau_ff + kp * (q_des + dof_pos_comp - self.dof_pos) + kd * (qd_des - self.dof_vel), -tau_max, tau_max)

        # torques[:] = tt
        # print(f"torque_estimate: {torques[0]}")
        # print(f"torques1: {torques1[0]}")
        # print(f"tau_ff: {tau_ff[0]}")

        # torques = torch.clip(tau_ff + self.Kp * (dof_pos_desired - self.dof_pos) + self.Kd * (joint_vel_des - self.dof_vel), -tau_max, tau_max)
        # tmp_max_torque = torch.clip(k * (self.dof_vel - v_max), 0, tau_max)
        # tmp_min_torque = torch.clip(k * (self.dof_vel + v_max), -tau_max, 0)
        # torques[:] = torch.where(self.dof_vel > tau_max / k + v_max,
        #                          torch.clip(torques, -tau_max * torch.ones_like(torques), tmp_max_torque), torques)
        # torques[:] = torch.where(self.dof_vel < -(tau_max / k + v_max),
        #                          torch.clip(torques, tmp_min_torque, tau_max * torch.ones_like(torques)), torques)
        # print("torques: ", torques)
        # torques = torch.zeros_like(torques)
        torques *= self.motor_not_broken_flag
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        self.torques[:] = torques.view(self.torques.shape)
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.extra_force),
                                                gymtorch.unwrap_tensor(self.extra_torque), gymapi.ENV_SPACE)

        # self.record_state()
        # self.torques_square_accumulated += torch.square(torques)
        # self.record_state_test()
        self.record_state_tensor_test()

    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        self.gait_commands_count += 1
        # print(f"gait_count-progress: {(self.gait_commands_count==self.progress_buf).all()}")
        self.schedule_random()
        # if self.push_flag and self.common_step_counter % self.push_interval == 0:  ### wsh_annotation: self.push_interval > 0
        #     self.push_robots()

        # # prepare quantities
        # self.base_quat[:] = self.root_states[:, 3:7]
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.update_pre_state()
        # self.record_states_into_buffer()
        self.record_states_duration_sim_dt()

        self.ground_height_current[:] = self.get_heights_xy(self.root_states[:, :2]) - self.init_position_bias_rel_world[:, 2]  # controller world frame
        self.update_body_trajectory()

        self.calculate_vel_horizon_frame(self.base_lin_vel, self.base_ang_vel, self.base_lin_vel_command,
                                         self.base_ang_vel_command, self.base_quat_horizon, vel_weight=0.8)
        self.calculate_ref_foot_xy(self.gait_phase_normed_leg_broken, self.ref_lin_vel_horizon_feet[..., :2], self.gait_periods, self.gait_duty)

        # self.modify_vel_command()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # if len(env_ids) > 0:
        #     # self.reset_idx(env_ids)
        #     # print(self.contact_forces[:, self.base_index, :])
        #     print(self.euler_xyz)
        #     time.sleep(100)

        # check target reach
        if self.custom_origins:
            self.check_target_reached()

        # modify vel command
        # self.modify_vel_command()
        self.compute_observations()

        if self.push_flag:  ### wsh_annotation: self.push_interval > 0
            # self.push_robots()
            self.push_robots_indexed()

        ### wsh_annotation
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]

        self.gait_params_act_last[:] = self.gait_params_act[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # draw height lines
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
            sphere_geom_target = gymutil.WireframeSphereGeometry(0.05, 32, 32, None, color=(0, 0, 1))
            sphere_geom_foot_contact = gymutil.WireframeSphereGeometry(0.03, 32, 32, None, color=(0, 1, 0))
            sphere_geom_foot_contact_err = gymutil.WireframeSphereGeometry(0.036, 16, 16, None, color=(0, 0, 1), color2=(1, 0, 0))

            if not self.add_terrain_obs:
                self.measured_heights = self.get_heights()

            if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
                self.target_points[:] = self.root_states[:, :3]
            target_points_z = self.get_heights_xy(self.target_points[:, :2])

            # feet contact indices
            contact_state_err = torch.abs(self.feet_contact_state - self.ref_phase_contact_state)
            # feet_contact_indices = torch.where(self.feet_contact_state > 0.5)
            # feet_contact_err_indices = torch.where(contact_state_err > 0.5)
            # position_contact = (self.feet_position_world.reshape(self.num_envs, 4, 3))[feet_contact_indices]
            # position_contact_err = self.feet_position_world.reshape(self.num_envs, 4, 3)[feet_contact_err_indices]

            for i in range(self.num_envs):
                # base_pos = (self.root_states[i, :3])#.cpu().numpy()
                # heights = self.measured_heights[i]#.cpu().numpy()
                # height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                #                                self.height_points[i])#.cpu().numpy()
                # for j in range(heights.shape[0]):
                #     x = height_points[j, 0] + base_pos[0]
                #     y = height_points[j, 1] + base_pos[1]
                #     z = heights[j]
                #     sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                #     gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                #
                # # draw target
                # pose = gymapi.Transform(gymapi.Vec3(self.target_points[i, 0], self.target_points[i, 1], target_points_z[i]), r=None)
                # gymutil.draw_lines(sphere_geom_target, self.gym, self.viewer, self.envs[i], pose)

                # draw feet contact
                feet_contact_indices = torch.where(self.feet_contact_state[i] > 0.5)
                feet_contact_err_indices = torch.where(contact_state_err[i] > 0.5)
                position_contact = (self.feet_position_world[i].reshape(4, 3))[feet_contact_indices]
                position_contact_err = self.feet_position_world[i].reshape(4, 3)[feet_contact_err_indices]
                for k in range(position_contact.shape[0]):
                    pose = gymapi.Transform(gymapi.Vec3(position_contact[k, 0], position_contact[k, 1], position_contact[k, 2]), r=None)
                    gymutil.draw_lines(sphere_geom_foot_contact, self.gym, self.viewer, self.envs[i], pose)
                for k in range(position_contact_err.shape[0]):
                    pose = gymapi.Transform(gymapi.Vec3(position_contact_err[k, 0], position_contact_err[k, 1], position_contact_err[k, 2]), r=None)
                    # gymutil.draw_lines(sphere_geom_foot_contact_err, self.gym, self.viewer, self.envs[i], pose)

            # draw fault motor frame origin
            env_tuples_motor_fault = torch.where(self.motor_not_broken_flag < 0.5)
            env_ids_motor_fault = env_tuples_motor_fault[0]
            motor_fault_indices = env_tuples_motor_fault[1]
            if len(env_ids_motor_fault) > 0:
                for i in range(len(env_ids_motor_fault)):
                    env_id = env_ids_motor_fault[i]
                    motor_id = motor_fault_indices[i]
                    rigid_body_id = (motor_id // 3) * 4 + motor_id % 3 + 1
                    self.gym.set_rigid_body_color(self.envs[env_id], self.a1_handles[env_id], rigid_body_id, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0, 0))

            print('')


    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device,
                               requires_grad=False)  # 10-50cm on each side
        x = 0.1 * torch.tensor([-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device,
                               requires_grad=False)  # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights_xy(self, points_xy: torch.Tensor):
        shape = points_xy.shape
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(shape[:-1], device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        points_xy_flatten = points_xy.clone().reshape(-1, 2)
        points_xy_flatten += self.terrain.border_size
        points_xy_flatten = (points_xy_flatten / self.terrain.horizontal_scale).long()
        px = points_xy_flatten[:, 0]
        py = points_xy_flatten[:, 1]
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        height = self.height_samples[px, py]
        return height.view(shape[:-1]) * self.terrain.vertical_scale

    def get_heights(self, env_ids=None):
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points),
                                    self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.root_states[:, :3]).unsqueeze(1)

        # return self.get_heights_xy(points[:, :, :2])

        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]

        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
    #
    # # ### wsh_annotation: get observation value
    # # def get_base_lin_vel(self):
    # #     return self.base_lin_vel
    # #
    # # def get_base_ang_vel(self):
    # #     return self.base_ang_vel
    #
    # def record_state(self):
    #     record_path = "/home/wsh/Documents/record_data/imitation/imitation_mpc_data_walk_2.csv"
    #     if self.common_step_counter == 0:
    #         if os.path.exists(record_path):
    #             os.remove(record_path)
    #         with open(record_path, 'a+') as fp:
    #             fp.write("gait type: walk, duty: 0.5, phase offset: (0, 0.3, 0.5, 0.8)" + '\n')
    #             fp.write("(0 ~ 3): gait period(s)" + '\n')
    #             fp.write("(0 ~ 3): gait period(s)" + '\n')
    #             fp.write("(4 ~ 6): body linear velocity(m/s)" + '\n')
    #             fp.write("(7 ~ 9): body angular velocity(rad/s)" + '\n')
    #             fp.write("(10~13): body quaternion(x, y, z, w)" + '\n')
    #             fp.write("(14~25): joint position(rad)" + '\n')
    #             fp.write("(26~37): joint velocities(rad/s)" + '\n')
    #             fp.write("(38~41): feet contact state(1->contact)" + '\n')
    #             fp.write("(42~44): velocity command(vx, vy, omega)" + '\n')
    #             fp.write("(45~56): torque command(Nm)" + '\n\n')
    #
    #     data = torch.cat((self.gait_period_offset + 0.5,
    #                       self.base_lin_vel,
    #                       self.base_ang_vel,
    #                       self.base_quat,
    #                       self.dof_pos,
    #                       self.dof_vel,
    #                       self.feet_contact_state,
    #                       self.commands[:, :3],
    #                       self.torques), dim=-1).cpu().numpy()
    #     self.record_data = np.concatenate((self.record_data, data), axis=0)
    #
    #     if self.common_step_counter >= 500 - 1:
    #         with open(record_path, 'a+') as fp:
    #             np.savetxt(fp, self.record_data, delimiter=",")
    #         exit(0)

    def record_state_test(self):
        if self.common_step_counter == 0 and self.record_path == '':
            record_dir = os.path.join('/home/wsh/Documents/pyProjects/IsaacGymEnvs/isaacgymenvs/runs', 'A1Limited_2024-05-29_01-52-33(obs+flag+height+jointVel_noGait)/record_test_data')
            os.makedirs(record_dir, exist_ok=True)
            # file_name = generate_filename('record_data_v-0_3_0')
            file_name = 'record_data-A1Limited_2024-05-29_01-52-33(obs+flag+height+jointVel_noGait).csv'
            self.record_path = os.path.join(record_dir, file_name)
            with open(self.record_path, 'a+') as fp:
                fp.write("[0:3]:     body position(m)" + '\n')
                fp.write("[3:7]:     body quaternion(x, y, z, w)" + '\n')
                fp.write("[7:10]:    body linear velocity(m/s)" + '\n')
                fp.write("[10:13]:   body angular velocity(rad/s)" + '\n')
                fp.write("[13:25]:   joint position(rad)" + '\n')
                fp.write("[25:37]:   joint velocities(rad/s)" + '\n')
                fp.write("[37:49]:   feet position world(m)" + '\n')
                fp.write("[49:61]:   feet velocity world(m/s)" + '\n')
                fp.write("[61:73]:   feet position body(m)" + '\n')
                fp.write("[73:85]:   feet velocity body(m/s)" + '\n')
                fp.write("[85:97]:   feet contact force(N)" + '\n')
                fp.write("[97:101]:  feet contact state(1->contact)" + '\n')
                fp.write("[101:104]: velocity command(vx, vy, omega)" + '\n')
                fp.write("[104:116]: torque command(Nm)" + '\n')
                fp.write("number of actions: " + str(self.num_actions) + '\n\n')

        data = torch.cat((self.root_states[:, :3],
                          self.base_quat,
                          self.base_lin_vel,
                          self.base_ang_vel,
                          self.dof_pos,
                          self.dof_vel,
                          self.feet_position_world,
                          self.feet_lin_vel_world,
                          self.feet_position_body,
                          self.feet_lin_vel_body,
                          self.feet_force,
                          self.feet_contact_state,
                          self.commands[:, :3],
                          self.torques,
                          self.last_actions_raw), dim=-1).cpu().numpy()
        self.record_data_test = np.concatenate((self.record_data_test, data), axis=0)

        if self.common_step_counter >= 1000:
            with open(self.record_path, 'a+') as fp:
                np.savetxt(fp, self.record_data_test, delimiter=",")
            sys.exit()

    def record_state_tensor_test(self):
        if self.common_step_counter == 0 and self.record_path == '':
            record_dir = os.path.join('/home/wsh/Documents/pyProjects/IsaacGymEnvs/isaacgymenvs/runs', 'A1Limited_2024-06-01_10-00-54/record_test_data')
            os.makedirs(record_dir, exist_ok=True)
            # file_name = generate_filename('record_data_v-0_3_0')
            file_name = 'save_tensor-A1Limited_2024-06-01_10-00-54_adaptive'
            self.record_path = os.path.join(record_dir, file_name+'.pt')
            print(f"save path: {self.record_path}")
            tmp_data = torch.cat((self.root_states[:, :3]-self.init_position_bias_rel_world,
                                  self.base_quat,
                                  self.base_lin_vel,
                                  self.base_ang_vel,
                                  self.dof_pos,
                                  self.dof_vel,
                                  self.feet_position_world-self.init_position_bias_rel_world.repeat(1, 4),
                                  self.feet_lin_vel_world,
                                  self.feet_position_body,
                                  self.feet_lin_vel_body,
                                  self.feet_force,
                                  self.feet_contact_state,
                                  self.commands[:, :3],
                                  self.torques,
                                  self.last_actions_raw,
                                  self.motor_broken_table[:, 0].unsqueeze(1),
                                  self.reset_buf.unsqueeze(1),
                                  self.current_push_velocity,
                                  self.env_step_height.unsqueeze(1),
                                  self.init_position_bias_rel_world,
                                  self.feet_height_rel_ground
                                  ), dim=-1)
            self.current_data_to_save = tmp_data.clone()
            self.length_record_tensor = 500
            self.tensor_to_save = torch.zeros_like(self.current_data_to_save).unsqueeze(1).expand(-1, self.length_record_tensor, -1).clone()

            widgets = ['Progress: ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer(), ' ',
                       progressbar.ETA(), ' ']
            self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=self.length_record_tensor+1).start()
        else:
            if self.common_step_counter >= self.length_record_tensor:
                data_dict = {
                    'tensor': self.tensor_to_save,
                    'description': data_description0
                }
                torch.save(data_dict, self.record_path)
                self.pbar.finish()
                sys.exit()
            self.current_data_to_save[:] = torch.cat((self.root_states[:, :3]-self.init_position_bias_rel_world,
                                                      self.base_quat,
                                                      self.base_lin_vel,
                                                      self.base_ang_vel,
                                                      self.dof_pos,
                                                      self.dof_vel,
                                                      self.feet_position_world-self.init_position_bias_rel_world.repeat(1, 4),
                                                      self.feet_lin_vel_world,
                                                      self.feet_position_body,
                                                      self.feet_lin_vel_body,
                                                      self.feet_force,
                                                      self.feet_contact_state,
                                                      self.commands[:, :3],
                                                      self.torques,
                                                      self.last_actions_raw,
                                                      self.motor_broken_table[:, 0].unsqueeze(1),
                                                      self.reset_buf.unsqueeze(1),
                                                      self.current_push_velocity,
                                                      self.env_step_height.unsqueeze(1),
                                                      self.init_position_bias_rel_world,
                                                      self.feet_height_rel_ground
                                                      ), dim=-1)
        self.tensor_to_save[:, self.common_step_counter, :] = self.current_data_to_save
        self.pbar.update(self.common_step_counter+1)

    def update_pre_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_mass_matrix_tensors(self.sim)

        self.simulate_counter += 1

        self.feet_lin_momentum[:] += self.est_feet_force * self.sim_dt
        self.ref_feet_lin_momentum[:] += self.ref_feet_force_mpc * self.sim_dt
        self.feet_ang_momentum[:] += torch.cross(self.feet_position_body_in_world_frame.reshape(self.num_envs, 4, 3), (self.est_feet_force * self.sim_dt).reshape(self.num_envs, 4, 3), dim=-1).reshape(self.num_envs, 12)
        self.ref_feet_ang_momentum[:] += torch.cross(self.feet_position_body_in_world_frame.reshape(self.num_envs, 4, 3),
                                                 (self.ref_feet_force_mpc * self.sim_dt).reshape(self.num_envs, 4, 3),
                                                 dim=-1).reshape(self.num_envs, 12)

        self.base_quat[:] = self.root_states[:, 3:7]
        self.euler_xyz[:] = get_euler_xyz2(self.base_quat)

        tmp_q = self.dof_pos.clone().reshape(self.num_envs, 4, 3)
        self.kinematic_feet_pos[:], self.jacobian[:], self.inverse_jacobian[:] = self.robot_kinematics.forward_kinematics(tmp_q)

        # self.calculate_ref_timing_phase()

        # self.update_horizon_quat()
        # print(f"base_quat_horizon: {self.base_quat_horizon[0]}")
        self.base_quat_horizon[:] = quat_from_euler_xyz(self.euler_xyz[:, 0], self.euler_xyz[:, 1], torch.zeros_like(self.euler_xyz[:, 2]))
        self.horizon_quat_in_world[:] = quat_from_euler_xyz(torch.zeros_like(self.euler_xyz[:, 0]),
                                                            torch.zeros_like(self.euler_xyz[:, 1]),
                                                            self.euler_xyz[:, 2])
        self.horizon_lin_vel[:] = quat_rotate_inverse(self.horizon_quat_in_world, self.root_states[:, 7:10])
        self.horizon_ang_vel[:] = quat_rotate_inverse(self.horizon_quat_in_world, self.root_states[:, 10:13])
        # print(f"base_quat_horizon1: {self.base_quat_horizon[0]}")
        # print("")
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # print(f"ang: {self.euler_xyz[0, :2] / torch.pi * 180.0}")
        # print(f"angVel: {self.base_ang_vel[0, :2] / torch.pi * 180.0}")
        # print(f"pro: {self.projected_gravity[0, :2]}")

        self.dof_pos_rel_init[:] = self.dof_pos - self.default_dof_pos

        self.feet_position_world[:] = self.rigid_body_states_reshape[:, self.feet_indices, 0:3].view(self.num_envs, -1)
        self.feet_lin_vel_world[:] = self.rigid_body_states_reshape[:, self.feet_indices, 7:10].view(self.num_envs, -1)

        for i in range(len(self.feet_indices)):
            self.feet_position_body_in_world_frame[:, i * 3: i * 3 + 3] = self.feet_position_world[:, i * 3: i * 3 + 3] - self.root_states[:, 0:3]
            self.feet_position_body[:, i * 3: i * 3 + 3] = quat_rotate_inverse(self.base_quat,
                                                                                     self.feet_position_world[:,
                                                                                     i * 3: i * 3 + 3] - self.root_states[
                                                                                                         :, 0:3])
            self.feet_lin_vel_body[:, i * 3: i * 3 + 3] = quat_rotate_inverse(self.base_quat,
                                                                                    self.feet_lin_vel_world[:,
                                                                                    i * 3: i * 3 + 3] - self.root_states[
                                                                                                        :, 7:10])
        self.feet_position_hip[:] = self.feet_position_body - self.hip_position_rel_body


        # self.feet_position_hip[:] = self.kinematic_feet_pos.reshape(self.num_envs, 12)
        # tmp_dq = self.dof_vel.clone().reshape(self.num_envs, 4, 3)
        # self.feet_lin_vel_body[:] = torch.matmul(self.jacobian, tmp_dq.unsqueeze(-1)).squeeze(-1).reshape(self.num_envs, 12)
        # self.feet_position_body[:] = self.feet_position_hip + self.hip_position_rel_body

        self.feet_position_hip_from_joint[:] = self.kinematic_feet_pos.reshape(self.num_envs, 12)
        tmp_dq = self.dof_vel.clone().reshape(self.num_envs, 4, 3)
        self.feet_lin_vel_body_from_joint[:] = torch.matmul(self.jacobian, tmp_dq.unsqueeze(-1)).squeeze(-1).reshape(self.num_envs, 12)

        self.feet_position_moved_hip[:] = self.feet_position_hip - self.leg_bias_rel_hip

        for i in range(len(self.feet_indices)):
            self.feet_position_hip_horizon_frame[:, i * 3: i * 3 + 3] = quat_rotate(self.base_quat_horizon, self.feet_position_hip[:, i * 3: i * 3 + 3])
            self.feet_velocity_hip_horizon_frame[:, i * 3: i * 3 + 3] = quat_rotate(self.base_quat_horizon,
                                                                                    self.feet_lin_vel_body[:,
                                                                                    i * 3: i * 3 + 3])
            self.feet_position_moved_hip_horizon_frame[:, i * 3: i * 3 + 3] = quat_rotate(self.base_quat_horizon, self.feet_position_moved_hip[:, i * 3: i * 3 + 3])

        self.feet_force[:] = self.contact_forces[:, self.feet_indices].view(self.num_envs, -1)
        self.feet_contact_state[:] = self.contact_forces[:, self.feet_indices, 2] > self.stance_foot_force_threshold
        self.feet_contact_state_obs[:] = self.feet_contact_state.float() - 0.5

        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, ((self.root_states[:, 7:10] - self.last_base_lin_vel_rel_world) / self.sim_params.dt - self.gravity_acc))
        self.last_base_lin_vel_rel_world[:] = self.root_states[:, 7:10].clone().detach()

        self.controller_reset_buf[:] = 0

        # self.base_lin_vel_error_square_accumulated += torch.square(self.tmp_lin_vel_command - self.base_lin_vel)
        # self.base_ang_vel_error_square_accumulated += torch.square(self.tmp_ang_vel_command - self.base_ang_vel)

        # self.record_states_into_buffer()

        # print("lin acc: ", self.base_lin_acc)
        # print("last vel:", self.last_base_lin_vel_rel_world)
        # print(f"body_height_real: {(self.root_states[0, 2] - 0.3) * 1000} mm")
        # print(f"base_lin_vel: {self.base_lin_vel[0]}")
        # print(f"base_ang_vel: {self.base_ang_vel[0]}")
        # print(f"torque: {self.torques[0, 9:]}")
        # print(f"joint_pos: {self.dof_pos[0]}")
        # print(f"joint_vel: {self.dof_vel[0]}")
        # print(f"foot_pos: {self.feet_position_hip[0]}")
        # print(f"foot_vel: {self.feet_lin_vel_body[0]}")
        # print("feet force: ", torch.sum(self.feet_force[0, [2, 5, 8, 11]]))
        # print(f"est feet forcet: {self.est_feet_force[0]}")
        # print(f"ref feet force mpc: {self.ref_feet_force_mpc[0]}")
        # print(f"est feet forcet error: {self.est_feet_force[0] - self.ref_feet_force_mpc[0]}")
        # print("--------------------------------------------------------------")

    def update_horizon_quat(self, env_ids=None):
        if env_ids is None:
            self.world2base_quat[:] = self.base_quat.clone()
            self.world2base_quat[:, :3] = -self.world2base_quat[:, :3].clone()
            self.horizon_quat_in_world[:] = quat_from_euler_xyz(torch.zeros_like(self.euler_xyz[:, 0]),
                                                                torch.zeros_like(self.euler_xyz[:, 1]),
                                                                self.euler_xyz[:, 2])
            self.horizon_quat_in_base[:] = quat_mul(self.world2base_quat, self.horizon_quat_in_world)
            self.base_quat_horizon[:] = self.horizon_quat_in_base.clone()
            self.base_quat_horizon[:, :3] = -self.base_quat_horizon[:, :3].clone()
        else:
            self.world2base_quat[env_ids] = self.base_quat[env_ids].clone()
            self.world2base_quat[env_ids, :3] = -self.world2base_quat[env_ids, :3].clone()
            self.horizon_quat_in_world[env_ids] = quat_from_euler_xyz(torch.zeros_like(self.euler_xyz[env_ids, 0]),
                                                                torch.zeros_like(self.euler_xyz[env_ids, 1]),
                                                                self.euler_xyz[env_ids, 2])
            self.horizon_quat_in_base[env_ids] = quat_mul(self.world2base_quat[env_ids],
                                                          self.horizon_quat_in_world[env_ids])
            self.base_quat_horizon[env_ids] = self.horizon_quat_in_base[env_ids].clone()
            self.base_quat_horizon[env_ids, :3] = -self.base_quat_horizon[env_ids, :3].clone()

    def update_ref_body_trajectory(self):
        self.vel_commands_body[:, :2] = self.commands[:, :2]
        self.vel_commands_body[:, 2] = 0.0
        self.vel_commands_body[:, 3:5] = 0.0
        self.vel_commands_body[:, 5] = self.commands[:, 2]

        self.vel_commands_world[:, :3] = quat_rotate(self.base_quat, self.vel_commands_body[:, :3])
        self.vel_commands_world[:, 3:] = quat_rotate(self.base_quat, self.vel_commands_body[:, 3:])
        self.vel_commands_world[self.env_index_plane, 3:] = self.vel_commands_body[self.env_index_plane, 3:]
        self.calculate_Rw_matrix(self.euler_xyz[:, 1:3], self.Rw_matrix)
        self.delta_theta[:] = torch.matmul(self.Rw_matrix, self.vel_commands_world[:, 3:].unsqueeze(-1)).squeeze(-1)
        self.delta_theta[self.env_index_plane, :] = self.vel_commands_world[self.env_index_plane, 3:]
        yaw_angle_turned = self.dt * self.delta_theta[:, 2]

        vxy_normalized = normalize(self.vel_commands_world[:, :2])
        yaw_turn_indices = torch.where(torch.abs(self.delta_theta[:, 2]) > 1.0e-4)[0].to(torch.long)
        not_yaw_turn_indices = torch.where(torch.abs(self.delta_theta[:, 2]) <= 1.0e-4)[0].to(torch.long)
        self.ref_body_trajectory[yaw_turn_indices, :2] += vec_rotate_z(vxy_normalized[yaw_turn_indices], yaw_angle_turned[yaw_turn_indices] / 2.0) * (2 * torch.sin(yaw_angle_turned[yaw_turn_indices] / 2.0) * self.vel_commands_world[yaw_turn_indices, :2].norm(p=2, dim=-1) / self.delta_theta[yaw_turn_indices, 2]).unsqueeze(-1)
        self.ref_body_trajectory[not_yaw_turn_indices, :2] += self.dt * self.vel_commands_world[not_yaw_turn_indices, :2]  # controller world frame
        # self.ref_body_trajectory[:, :2] += self.act_body_trajectory[:, :2]  # controller world frame
        self.ref_body_trajectory[:, 2] = self.height_commands[:, 0] + self.ground_height_current  # controller world frame

        self.ref_body_trajectory[:, 3:5] = self.euler_xyz[:, :2]  # controller world frame
        self.ref_body_trajectory[self.env_index_plane, 3:5] = 0.0
        # env_ids = (self.delta_theta[:, 2] > 1.0e-3).nonzero(as_tuple=False).flatten()
        # self.ref_body_trajectory[env_ids, 5] += self.dt * self.delta_theta[env_ids, 2]  # controller world frame
        # env_ids = (torch.any(self.delta_theta, dim=1) > 1.0e-3).nonzero(as_tuple=False).flatten()
        self.ref_body_trajectory[:, 5] = wrap_to_pi(self.ref_body_trajectory[:, 5] + yaw_angle_turned)  # controller horizon frame

        self.ref_body_trajectory[:, 6:8] = vec_rotate_z(self.vel_commands_world[:, :2], yaw_angle_turned)  # controller world frame
        self.ref_body_trajectory[:, 8] = 0.0  # body frame
        self.ref_body_trajectory[:, 9:12] = self.vel_commands_body[:, 3:]  # body frame

        # print(f"vel_commands_world: {self.vel_commands_world[3, :2]}")
        # print(f"delta_theta: {self.delta_theta[0]}")
        # print(f"yaw_rate_commands: {self.commands[0, 2]}")
        # print(f"ref_body_trajectory: {self.ref_body_trajectory[3, 5]}")
        # print(f"dt: {self.dt}")
        # print(f"progress_buf: {self.progress_buf[:]}")

    def update_body_trajectory(self):
        self.act_body_trajectory[:, :3] = self.root_states[:, :3] - self.init_position_bias_rel_world
        self.act_body_trajectory[:, 3:6] = self.euler_xyz.clone()
        self.act_body_trajectory[:, 6:8] = self.root_states[:, 7:9]
        self.act_body_trajectory[:, 8] = self.base_lin_vel[:, 2]
        self.act_body_trajectory[:, 9:12] = self.base_ang_vel.clone()  # body frame

        self.body_traj_error[:] = self.ref_body_trajectory - self.act_body_trajectory
        self.body_traj_error[:, 5] = wrap_to_pi(self.body_traj_error[:, 5])
        # self.ref_body_trajectory[:, :2] = self.act_body_trajectory[:, :2]
        # self.ref_body_trajectory[:, 5] = self.act_body_trajectory[:, 5]

        xy_position_error_threshold = 0.1
        xy_position_error = torch.clip(self.ref_body_trajectory[:, :2] - self.act_body_trajectory[:, :2], -xy_position_error_threshold, xy_position_error_threshold)
        self.ref_body_trajectory[:, :2] = self.act_body_trajectory[:, :2] + xy_position_error

        yaw_angle_error_threshold = 0.1
        yaw_angle_error = torch.clip(self.ref_body_trajectory[:, 5] - self.act_body_trajectory[:, 5], -yaw_angle_error_threshold, yaw_angle_error_threshold)
        self.ref_body_trajectory[:, 5] = self.act_body_trajectory[:, 5] + yaw_angle_error

        # self.body_traj_error[:] = self.ref_body_trajectory - self.act_body_trajectory

        # print(f"act_body_trajectory: {self.act_body_trajectory[3, 5]}")
        # idx = torch.where(self.body_traj_error[:, 5]>1.0)[0].to(torch.long)
        # if len(idx):
        #     print(f"body_traj_error: {self.body_traj_error[idx, 5]}")
        #     print(f"wrapped_body_traj_error: {wrap_to_pi(self.body_traj_error[idx, 5])}")

    def update_motor_broken_state(self):
        env_ids1 = torch.where(self.progress_buf == self.motor_broken_count1)[0]
        # env_ids2 = torch.where(self.progress_buf == self.motor_broken_count2)[0]
        self.motor_not_broken_flag[env_ids1] = self.motor_not_broken_flag1[env_ids1].clone()
        # self.motor_not_broken_flag[env_ids2] = self.motor_not_broken_flag2[env_ids2].clone()
        self.leg_broken_flag[env_ids1] = self.leg_broken_flag1[env_ids1].clone()
        # self.leg_broken_flag[env_ids2] = self.leg_broken_flag2[env_ids2].clone()
        self.leg_not_broken_flag[env_ids1] = (~self.leg_broken_flag[env_ids1].bool()).to(torch.long)
        # self.leg_not_broken_flag[env_ids2] = (~self.leg_broken_flag[env_ids2].bool()).to(torch.long)
        self.leg_broken_count[env_ids1] = torch.sum(self.leg_broken_flag[env_ids1], dim=1)
        self.gait_index_leg_broken[env_ids1] = self.index_leg_broken_flag_to_gait[self.leg_broken_flag[env_ids1].transpose(-2, -1).unbind(0)]
        self.gait_params_leg_broken[env_ids1] = self.gait_tensor_leg_broken[self.gait_index_leg_broken[env_ids1]]
        # self.leg_broken_count[env_ids2] = torch.sum(self.leg_broken_flag[env_ids2], dim=1)
        # self.gait_index_leg_broken[env_ids2] = self.index_leg_broken_flag_to_gait[self.leg_broken_flag[env_ids2].transpose(-2, -1).unbind(0)]
        # self.gait_params_leg_broken[env_ids2] = self.gait_tensor_leg_broken[self.gait_index_leg_broken[env_ids2]]
        # print(((~self.gait_params_leg_broken[:, -4:].bool()).to(torch.long) == self.leg_broken_flag).all())
        # print(f"motor_not_broken_flag: {self.motor_not_broken_flag[:]}")

    def update_timing_phase_leg_broken(self):
        self.gait_phase_leg_broken[:] = (-self.gait_params_leg_broken[:, 2:6] + self.gait_params_leg_broken[:, 6:10] * ((self.progress_buf * self.dt) / self.gait_params_leg_broken[:, 0]).unsqueeze(-1)) % 1.0
        self.gait_phase_normed_leg_broken[:] = torch.where(self.gait_phase_leg_broken <= self.gait_params_leg_broken[:, 1].unsqueeze(-1), 0.5 * self.gait_phase_leg_broken / self.gait_params_leg_broken[:, 1].unsqueeze(-1), 0.5 + 0.5 * (self.gait_phase_leg_broken - self.gait_params_leg_broken[:, 1].unsqueeze(-1)) / (1. - self.gait_params_leg_broken[:, 1].unsqueeze(-1)))
        self.ref_phase_C_des_leg_broken[:] = self.ref_phase_trans_distribution.cdf(self.gait_phase_normed_leg_broken) * (1 - self.ref_phase_trans_distribution.cdf(self.gait_phase_normed_leg_broken - 0.5)) + self.ref_phase_trans_distribution.cdf(self.gait_phase_normed_leg_broken - 1)
        self.ref_phase_norm_sincos_leg_broken[:, [0, 2, 4, 6]] = torch.sin(self.gait_phase_normed_leg_broken * 2.0 * torch.pi)
        self.ref_phase_norm_sincos_leg_broken[:, [1, 3, 5, 7]] = torch.cos(self.gait_phase_normed_leg_broken * 2.0 * torch.pi)
        # print(f"ref_phase_C_des_leg_broken: {self.ref_phase_C_des_leg_broken[0]}")
        # print(f"ref_phase_C_des: {self.ref_phase_C_des[0]}")
        # print(f"ref_phase_norm_sincos_leg_broken: {self.ref_phase_norm_sincos_leg_broken[0]}")

    def update_global_clock(self):
        self.global_clock_phase[:] = ((self.progress_buf * self.dt) / self.global_clock_period) % 1.0
        self.global_clock_phase_pi[:] = self.global_clock_phase * 2.0 * torch.pi
        self.global_clock_phase_sin_cos[:, 0] = torch.sin(self.global_clock_phase_pi)
        self.global_clock_phase_sin_cos[:, 1] = torch.cos(self.global_clock_phase_pi)
        # print(f"global_clock_phase: {self.global_clock_phase[0]}")

    def modify_vel_command2(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 3] = 0.
        self.commands[:, 2] = self._heading_to_omega(heading)

    def modify_vel_command(self):
        env_ids = ((self.progress_buf > 0) & ((self.progress_buf == 25) | (self.progress_buf % self.commands_change_random_count == 0))).nonzero(as_tuple=False).flatten()
        self.commands[env_ids, 0] = torch_rand_float(self.scheduled_command_x_range[0], self.scheduled_command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.scheduled_command_y_range[0], self.scheduled_command_y_range[1],
                                                     (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(self.scheduled_command_yaw_range[0], self.scheduled_command_yaw_range[1],
                                                     (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 2] = self.commands[env_ids, 3].clone()

        # if self.common_step_counter % 120 > 60:
        #     self.commands[:, 3] = 3
        # else:
        #     self.commands[:, 3] = 0

        self.commands_heading_flag[env_ids] = torch.randint_like(self.commands_heading_flag[env_ids], low=0, high=2)
        env_ids_heading = (self.commands_heading_flag > 0.5).nonzero(as_tuple=False).flatten()
        env_ids_not_heading = (self.commands_heading_flag < 0.5).nonzero(as_tuple=False).flatten()
        self.commands[env_ids_not_heading, 2] = self.commands[env_ids_not_heading, 3]
        # calculate omega command
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[env_ids_heading, 2] = self._heading_to_omega(heading)[env_ids_heading]
        # print(self.commands[0, 2])

        # self.commands[env_ids, 2] = self.commands[env_ids, 3].clone()

        # self.commands[:, 0] = 0.5
        # self.commands[:1200, 0] = -0.5
        # self.commands[1200:2400, 0] = 0.
        # self.commands[2400:3600, 0] = 0.5
        # self.commands[3600:4800, 0] = 1.0
        # self.commands[4800:6000, 0] = 1.5
        # self.commands[:, 1] = 0.
        # self.commands[6000:, 1] = 0.5
        # self.commands[:, 3] = 0.
        # self.commands[:, 2] = self._heading_to_omega(heading)

        # self.commands[:, 2] = 0.
        # print(f"heading_vec: {forward[0]}")
        # print(f"heading: {heading[0]}")
        # print(f"target_vec: {self.target_pos_rel[0]}")
        # print(f"target_yaw: {self.target_yaw[0]}")
        # print(f"omega: {self.commands[0, 2]}")
        # print("------------------------------------------------")


        # self.commands[:, 2] = 0.
        # if self.common_step_counter % 120 > 90:
        #     self.commands[:, 2] = 5.
        # else:
        #     self.commands[:, 2] = 0.

        self.commands[:] *= ~((self.commands[:, :3].abs() < self.xy_velocity_threshold).all(dim=-1)).unsqueeze(1)
        self.command_lin_vel_x[:] = self.commands[:, 0].unsqueeze(-1).clone()

        # env_ids2 = (self.progress_buf < 150).nonzero(as_tuple=False).flatten()
        # self.commands[env_ids2] = 0.
        self.commands[:, :3] = torch.clamp(self.commands[:, :3], self.commands_last - self.commands_delta, self.commands_last + self.commands_delta)
        # self.commands[:, 2] = 3.
        self.commands_last[:] = self.commands[:, :3]

        self.base_lin_vel_command[:, :2] = self.commands[:, :2].clone()
        self.base_ang_vel_command[:, 2] = self.commands[:, 2].clone()

        # count = (self.common_step_counter % 1500) // 200
        # if count == 0:
        #     self.commands[:, 0] = 0.5
        # elif count == 1:
        #     self.commands[:, 0] = 1.0
        # elif count == 2:
        #     self.commands[:, 0] = 1.5
        # elif count == 3:
        #     self.commands[:, 0] = 1.0
        # else:
        #     self.commands[:, 0] = 0.5
        #
        # self.commands[:, 1] = 0.0
        # self.commands[:, 2] = 0.0

        # n = 1500  # 3000 # 100
        # count = self.common_step_counter % n
        # if count > n // 2:
        #     count = n - count
        # self.commands[:, 0] = 0.004 * count  # 0.002 # 0.06
        # print(count)
        # self.commands[:, 0] = 0

        # self.commands[:, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (self.num_envs, 1),
        #                                        device=self.device).squeeze()
        # self.commands[:, 0] = 2
        # self.commands[:] *= ~((self.commands[:, :3].abs() < self.xy_velocity_threshold).all(dim=-1)).unsqueeze(1)
        # self.tmp_lin_vel_command = self.commands[:, :3].clone()
        # self.tmp_lin_vel_command[:, 2] = 0
        # self.tmp_ang_vel_command = self.commands[:, :3].clone()
        # self.tmp_ang_vel_command[:, :2] = 0

        # self.obs_buffer_dict["commands"].record(self.obs_name_to_value["commands"])

    def modify_desired_gait_command(self):
        env_ids = (self.progress_buf % self.gait_commands_change_random_count == 0).nonzero(as_tuple=False).flatten()
        # self.gait_commands[env_ids, 0] = torch_rand_float(self.command_gait_period_range[0],
        #                                                   self.command_gait_period_range[1],
        #                                                   (len(env_ids), 1),
        #                                                   device=self.device).squeeze()
        # self.gait_commands[env_ids, 1] = torch_rand_float(self.command_gait_duty_range[0],
        #                                                   self.command_gait_duty_range[1],
        #                                                   (len(env_ids), 1),
        #                                                   device=self.device).squeeze()
        # self.gait_commands[env_ids, 2:5] = torch_rand_float(self.command_gait_offset_range[0],
        #                                                     self.command_gait_offset_range[1],
        #                                                     (len(env_ids), 3),
        #                                                     device=self.device)  # [FL RR RL FR]
        # self.gait_commands[env_ids, 5] = 0.0

        # self.gait_commands[env_ids, 0] = 0.3
        # self.gait_commands[env_ids, 1] = 0.25
        # self.gait_commands[env_ids, 2] = 0.25
        # self.gait_commands[env_ids, 3] = 0.75
        # self.gait_commands[env_ids, 4] = 0.5

        gait_list_ids = torch.randint(0, self.num_gait, (len(env_ids),))
        gait_list_ids[:] = 1
        self.gait_id[env_ids] = gait_list_ids.to(self.device).clone()
        self.gait_commands[env_ids, :] = self.gait_tensor[gait_list_ids].clone()

        self.gait_periods[env_ids] = self.gait_commands[env_ids, 0]
        self.gait_duty[env_ids] = self.gait_commands[env_ids, 1]
        self.gait_stance_time[env_ids] = self.gait_periods[env_ids] * self.gait_commands[env_ids, 1]
        self.ref_delta_phase[env_ids] = self.dt / self.gait_periods[env_ids]
        self.ref_delta_phase_sim_step[env_ids] = self.sim_dt / self.gait_periods[env_ids]
        self.gait_commands_count[env_ids] = 0

        # ----------------------------------- specific commands -----------------------------------
        # if self.common_step_counter == 0:
        #     self.specific_gait_dict = {}
        #     self.specific_gait_dict["walk"] = torch.tensor([0.5, 0.75, 0.5, 0.75, 0.25, 0.25], device=self.device)
        #     self.specific_gait_dict["trot"] = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.0], device=self.device)
        #     self.specific_gait_dict["flying_trot"] = torch.tensor([0.5, 0.25, 0.5, 0.5, 0.0, 0.0], device=self.device)
        #     self.specific_gait_dict["pace"] = torch.tensor([0.3, 0.7, 0.5, 0.0, 0.5, 0.0], device=self.device)
        #     self.specific_gait_dict["bound"] = torch.tensor([0.3, 0.3, 0.0, 0.5, 0.5, 0.0], device=self.device)
        #     self.specific_gait_dict["jump"] = torch.tensor([0.2, 0.5, 0.0, 0.0, 0.0, 0.0], device=self.device)
        #     self.specific_gait_dict["transverse_gallop"] = torch.tensor([0.24, 0.3, 0.25, 0.6, 0.85, 0.0], device=self.device)
        #     self.specific_gait_dict["random"] = torch.tensor([0.6, 0.6, 0.37, 0.32, 0.6, 0.0], device=self.device)
        # self.gait_commands[:, :] = self.specific_gait_dict["trot"]
        # # if self.common_step_counter % 289 == 0:
        # #     # tmp_gaits = list(self.specific_gait_dict.values())
        # #     # index = torch.randint(0, len(tmp_gaits), (1,)).item()
        # #     # self.gait_commands[:, :] = tmp_gaits[index]
        # #     self.gait_commands[:, :] = self.specific_gait_dict["trot"]
        # #     self.gait_commands_count[:] = 0
        # #     # print("to transverse_gallop")
        # # if self.common_step_counter % 578 == 0:
        # #     self.gait_commands[:, :] = self.specific_gait_dict["trot"]
        # #     self.gait_commands_count[:] = 0
        # #     # print("to flying_trot")
        # self.gait_periods[:] = self.gait_commands[:, 0]
        # self.gait_duty[:] = self.gait_commands[:, 1]
        # self.gait_stance_time[:] = self.gait_periods * self.gait_commands[:, 1]
        # self.ref_delta_phase[:] = self.dt / self.gait_periods

    def modify_desired_height_command(self):
        env_ids = (self.progress_buf % self.height_commands_change_random_count == 0).nonzero(as_tuple=False).flatten()
        self.height_commands[env_ids] = torch_rand_float(self.scheduled_command_height_range[0], self.scheduled_command_height_range[1],
                                                         (len(env_ids), 1), device=self.device)

        # specific commands
        self.height_commands[:] = 0.3

    def record_states_duration_sim_dt(self):
        for key in self.record_items_duration_sim_dt:
            self.obs_buffer_dict[key].record(self.obs_name_to_value[key])

    def record_states_duration_learn_dt(self):
        for key in self.record_items_duration_learn_dt:
            self.obs_buffer_dict[key].record(self.obs_name_to_value[key])

    def record_states_into_buffer(self):
        ### wsh_annotation: record new states into buffer
        for key in self.record_items:
            if key != "commands" and key != "heightMeasurement" and key != "feet_phase_sincos" and key != "ref_phase_norm_sincos_current" and key != "ref_phase_norm_sincos_next" and key != "global_clock_phase_sin_cos":
                self.obs_buffer_dict[key].record(self.obs_name_to_value[key])

    def record_commands(self):
        self.obs_buffer_dict["commands"].record(self.obs_name_to_value["commands"])

    def record_ref_phase(self):
        self.obs_buffer_dict["ref_phase_norm_sincos_current"].record(self.obs_name_to_value["ref_phase_norm_sincos_current"])
        self.obs_buffer_dict["ref_phase_norm_sincos_next"].record(self.obs_name_to_value["ref_phase_norm_sincos_next"])

    def record_global_clock_phase(self):
        self.obs_buffer_dict["global_clock_phase_sin_cos"].record(self.obs_name_to_value["global_clock_phase_sin_cos"])

    def calculate_phase(self, sin_theta, cos_theta):
        # Calculate theta
        theta = torch.atan2(sin_theta.clone(), cos_theta.clone())
        # Adjust theta to the range [0, 2*pi]
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)
        phase = theta / (2 * torch.pi)
        return phase

    def schedule_random(self):
        if self.if_schedule_command:
            self.schedule_delta = self.common_step_counter / (24. * 400.)
            if self.schedule_delta > 1:
                self.schedule_delta = 1.
            self.commands_delta = self.schedule_delta * 3000. * self.dt
            # self.commands_delta = 0. * self.dt

            if self.common_step_counter == 24 * 0:
                self.scheduled_command_x_range[0] = -0.5
                self.scheduled_command_x_range[1] = 1.
                self.scheduled_command_y_range[0] = -0.6
                self.scheduled_command_y_range[1] = 0.6
                self.scheduled_command_yaw_range[0] = -1.
                self.scheduled_command_yaw_range[1] = 1.
                self.scheduled_push_velocity_range[0] = -0.5
                self.scheduled_push_velocity_range[1] = 0.5
                self.scheduled_command_height_range[0] = 0.29
                self.scheduled_command_height_range[1] = 0.31
            elif self.common_step_counter == 24 * 300:
                self.scheduled_command_x_range[0] = -1.
                self.scheduled_command_x_range[1] = 2.
                self.scheduled_command_y_range[0] = -1.
                self.scheduled_command_y_range[1] = 1.
                self.scheduled_command_yaw_range[0] = -2.
                self.scheduled_command_yaw_range[1] = 2.
                self.scheduled_push_velocity_range[0] = -1.
                self.scheduled_push_velocity_range[1] = 1.
                self.scheduled_command_height_range[0] = 0.27
                self.scheduled_command_height_range[1] = 0.33
            elif self.common_step_counter == 24 * 600:
                self.scheduled_command_x_range[0] = -1.2
                self.scheduled_command_x_range[1] = 3.
                self.scheduled_command_y_range[0] = -1.2
                self.scheduled_command_y_range[1] = 1.2
                self.scheduled_command_yaw_range[0] = -3.14
                self.scheduled_command_yaw_range[1] = 3.14
                self.scheduled_push_velocity_range[0] = -1.5
                self.scheduled_push_velocity_range[1] = 1.5
                self.scheduled_command_height_range[0] = 0.22
                self.scheduled_command_height_range[1] = 0.36
            # elif self.common_step_counter == 24 * 600:
            #     self.scheduled_command_x_range[0] = -0.6
            #     self.scheduled_command_x_range[1] = 1.1
            #     self.scheduled_command_y_range[0] = -1.0
            #     self.scheduled_command_y_range[1] = 1.0
            #     self.scheduled_command_yaw_range[0] = -2.
            #     self.scheduled_command_yaw_range[1] = 2.
            #     self.scheduled_push_velocity_range[0] = -1.
            #     self.scheduled_push_velocity_range[1] = 1.
            #     self.scheduled_command_height_range[0] = 0.25
            #     self.scheduled_command_height_range[1] = 0.34
            # elif self.common_step_counter == 24 * 800:
            #     self.scheduled_command_x_range[0] = -0.8
            #     self.scheduled_command_x_range[1] = 1.5
            #     self.scheduled_command_y_range[0] = -1.1
            #     self.scheduled_command_y_range[1] = 1.1
            #     self.scheduled_command_yaw_range[0] = -2.5
            #     self.scheduled_command_yaw_range[1] = 2.5
            #     self.scheduled_push_velocity_range[0] = -1.5
            #     self.scheduled_push_velocity_range[1] = 1.5
            #     self.scheduled_command_height_range[0] = 0.23
            #     self.scheduled_command_height_range[1] = 0.35
            # elif self.common_step_counter == 24 * 1200:
            #     self.scheduled_command_x_range[0] = -1.0
            #     self.scheduled_command_x_range[1] = 2.0
            #     self.scheduled_command_y_range[0] = -1.2
            #     self.scheduled_command_y_range[1] = 1.2
            #     self.scheduled_command_yaw_range[0] = -3.0
            #     self.scheduled_command_yaw_range[1] = 3.0
            #     self.scheduled_push_velocity_range[0] = -2.
            #     self.scheduled_push_velocity_range[1] = 2.
            #     self.scheduled_command_height_range[0] = 0.20
            #     self.scheduled_command_height_range[1] = 0.36
            # elif self.common_step_counter == 24 * 1600:
            #     self.scheduled_command_x_range[0] = -1.0
            #     self.scheduled_command_x_range[1] = 2.5
            #     self.scheduled_command_y_range[0] = -1.5
            #     self.scheduled_command_y_range[1] = 1.5
            #     self.scheduled_command_yaw_range[0] = -3.0
            #     self.scheduled_command_yaw_range[1] = 3.0
            #     self.scheduled_push_velocity_range[0] = -2.
            #     self.scheduled_push_velocity_range[1] = 2.
            #     self.scheduled_command_height_range[0] = 0.20
            #     self.scheduled_command_height_range[1] = 0.36
            # elif self.common_step_counter == 24 * 2000:
            #     self.scheduled_command_x_range[0] = -1.0
            #     self.scheduled_command_x_range[1] = 3.0
            #     self.scheduled_command_y_range[0] = -1.5
            #     self.scheduled_command_y_range[1] = 1.5
            #     self.scheduled_command_yaw_range[0] = -3.14
            #     self.scheduled_command_yaw_range[1] = 3.14
            #     self.scheduled_push_velocity_range[0] = -2.
            #     self.scheduled_push_velocity_range[1] = 2.
            #     self.scheduled_command_height_range[0] = 0.20
            #     self.scheduled_command_height_range[1] = 0.36
            # elif self.common_step_counter == 24 * 3000:
            #     self.scheduled_command_x_range[0] = -1.0
            #     self.scheduled_command_x_range[1] = 4.0
            #     self.scheduled_command_y_range[0] = -1.5
            #     self.scheduled_command_y_range[1] = 1.5
            #     self.scheduled_command_yaw_range[0] = -3.14
            #     self.scheduled_command_yaw_range[1] = 3.14
            #     self.scheduled_push_velocity_range[0] = -2.
            #     self.scheduled_push_velocity_range[1] = 2.
            #     self.scheduled_command_height_range[0] = 0.20
            #     self.scheduled_command_height_range[1] = 0.36
            # elif self.common_step_counter == 24 * 4000:
            #     self.scheduled_command_x_range[0] = -1.0
            #     self.scheduled_command_x_range[1] = 5.0
            #     self.scheduled_command_y_range[0] = -1.5
            #     self.scheduled_command_y_range[1] = 1.5
            #     self.scheduled_command_yaw_range[0] = -3.14
            #     self.scheduled_command_yaw_range[1] = 3.14
            #     self.scheduled_push_velocity_range[0] = -2.
            #     self.scheduled_push_velocity_range[1] = 2.
            #     self.scheduled_command_height_range[0] = 0.20
            #     self.scheduled_command_height_range[1] = 0.36
            # elif self.common_step_counter == 24 * 5000:
            #     self.scheduled_command_x_range[0] = -1.0
            #     self.scheduled_command_x_range[1] = 6.0
            #     self.scheduled_command_y_range[0] = -1.5
            #     self.scheduled_command_y_range[1] = 1.5
            #     self.scheduled_command_yaw_range[0] = -3.14
            #     self.scheduled_command_yaw_range[1] = 3.14
            #     self.scheduled_push_velocity_range[0] = -2.
            #     self.scheduled_push_velocity_range[1] = 2.
            #     self.scheduled_command_height_range[0] = 0.20
            #     self.scheduled_command_height_range[1] = 0.36

    # def gait_tracking(self, desired_gait, gait_params_act_raw_last, obs):
    #     obs_shaped = torch.cat([desired_gait, obs[:, :36], gait_params_act_raw_last, obs[:, 47:75]], dim=-1)
    #     with torch.no_grad():
    #         return self.gait_tracking_policy(obs_shaped.clone().detach()) * 0.25

    def gait_tracking(self):
        obs_shaped = torch.cat([self.gait_commands,
                                self.height_commands,
                                self.base_lin_vel,
                                self.base_ang_vel,
                                self.euler_roll,
                                self.euler_pitch,
                                self.commands[:, :3],
                                self.dof_pos,
                                self.dof_vel,
                                self.gait_params_act_raw_last,
                                self.feet_position_hip,
                                self.feet_lin_vel_body,
                                self.feet_contact_state_obs], dim=-1)
        with torch.no_grad():
            gait_tracked = self.gait_tracking_policy(obs_shaped.clone().detach())
            gait_tracked = torch.clip(gait_tracked, min=-4.0, max=4.0) * 0.25
            self.gait_params_act_raw_last = gait_tracked.clone()
            gait_tracked[:, :5] = (gait_tracked[:, :5] + 1.0) * 0.5
            return gait_tracked

    def restore_gait_tracking_policy(self, fn):
        from rl_games.algos_torch import torch_ext
        from collections import OrderedDict

        model_params = torch_ext.load_checkpoint(fn)['model']
        actor_net_strings = ['actor_mlp', 'mu']
        actor_net_params = OrderedDict()
        for key, value in model_params.items():
            for search_string in actor_net_strings:
                if search_string in key:
                    actor_net_params[key] = value
                    break
        self.gait_tracking_policy.load_state_dict(actor_net_params, strict=False)

        if self.normalize_gait_tracking_input:
            input_normalize_strings = ['running_mean_std']
            input_normalize_params = OrderedDict()
            for key, value in model_params.items():
                # Check if any of the search strings are in the key
                if any(item in key for item in input_normalize_strings):
                    # Iterate over all occurrences of the search strings in the key
                    for search_string in input_normalize_strings:
                        for i in range(len(key) - len(search_string) + 1):
                            # Extract the portion of the key value that contains the search string
                            substring = key[i:i + len(search_string)]
                            if substring == search_string and i + len(search_string) < len(key) - 1:
                                input_normalize_params[key[i + len(search_string) + 1:]] = value
                                break
            if input_normalize_params:
                self.gait_tracking_policy.running_mean_std.load_state_dict(input_normalize_params)
        print("gait tracking policy actor net initialized.")

    def calculate_ref_timing_phase(self):
        env_ids_init = (self.gait_commands_count == 0).nonzero(as_tuple=False).flatten()
        env_ids_running = (self.gait_commands_count > 0).nonzero(as_tuple=False).flatten()

        self.ref_phase_current[env_ids_init] = torch.cat((self.gait_commands[env_ids_init, 5].unsqueeze(-1), -self.gait_commands[env_ids_init, 2:5]+self.gait_commands[env_ids_init, 5].unsqueeze(-1)), dim=-1)[:, [1, 0, 3, 2]]  # [FR FL RR RL] -> [FL FR RL RR]
        self.ref_phase_current[env_ids_running] += self.ref_delta_phase[env_ids_running].unsqueeze(-1)
        # self.ref_phase_current[env_ids_running] += self.ref_delta_phase_sim_step[env_ids_running].unsqueeze(-1)
        self.ref_phase_current[:] %= 1.
        self.ref_phase_pi_current[:] = self.ref_phase_current * 2. * torch.pi

        duty = self.gait_commands[:, 1].unsqueeze(-1)
        self.ref_phase_norm_current[:] = torch.where(self.ref_phase_current<=duty, 0.5 * self.ref_phase_current/duty, 0.5 + 0.5 * (self.ref_phase_current - duty)/(1. - duty)) % 1
        # self.ref_phase_norm_current[:, 0] = 0.75  # three legs
        self.ref_phase_norm_pi_current[:] = self.ref_phase_norm_current * 2. * torch.pi
        self.ref_phase_norm_pi_next[:] = self.ref_phase_current + self.ref_delta_phase.unsqueeze(-1)
        # self.ref_phase_norm_pi_next[:] = self.ref_phase_current + self.ref_delta_phase_sim_step.unsqueeze(-1)
        self.ref_phase_norm_pi_next[:] %= 1.
        self.ref_phase_norm_pi_next[:] = torch.where(self.ref_phase_norm_pi_next<=duty, 0.5 * self.ref_phase_norm_pi_next/duty, 0.5 + 0.5 * (self.ref_phase_norm_pi_next - duty)/(1. - duty))
        self.ref_phase_norm_pi_next[:] *= 2. * torch.pi

        for i in range(4):
            self.ref_phase_sincos_current[:, 2 * i] = torch.sin(self.ref_phase_pi_current[:, i])
            self.ref_phase_sincos_current[:, 2 * i + 1] = torch.cos(self.ref_phase_pi_current[:, i])

            self.ref_phase_norm_sincos_current[:, 2 * i] = torch.sin(self.ref_phase_norm_pi_current[:, i])
            self.ref_phase_norm_sincos_current[:, 2 * i + 1] = torch.cos(self.ref_phase_norm_pi_current[:, i])

            self.ref_phase_norm_sincos_next[:, 2 * i] = torch.sin(self.ref_phase_norm_pi_next[:, i])
            self.ref_phase_norm_sincos_next[:, 2 * i + 1] = torch.cos(self.ref_phase_norm_pi_next[:, i])

        self.ref_phase_contact_state[:] = (self.ref_phase_norm_current <= 0.5).float()
        self.ref_phase_contact_num[:] = torch.sum(self.ref_phase_contact_state, dim=-1).unsqueeze(-1)

        self.calculate_C_des(self.ref_phase_norm_current)
        self.calculate_foot_pos_track_weight(self.ref_phase_norm_current)
        # print(f"ref_phase_norm_sincos_current: {self.ref_phase_norm_sincos_current[0]}")

    def calculate_ref_dof_commands(self):
        self.motion_planning_interface.update_body_vel_x_y_wz(self.commands[:, :3])
        self.motion_planning_interface.update_body_height_offset(self.height_commands - 0.3)

        gait_period_offset = (self.gait_commands[:, 0] - 0.5).unsqueeze(-1).repeat(1, 4)
        gait_duty_cycle_offset = (self.gait_commands[:, 1] - 0.5).unsqueeze(-1).repeat(1, 4)
        gait_phase_offset = self.gait_commands[:, 2:6].clone()
        gait_phase_offset[:, :2] -= 0.5
        gait_phase_offset[:, 3] = self.calculate_phase(self.ref_phase_sincos_current[:, 2], self.ref_phase_sincos_current[:, 3])
        self.motion_planning_interface.update_gait_planning(True, gait_period_offset, gait_duty_cycle_offset,
                                                            gait_phase_offset, None)
        # self.motion_planning_interface.change_gait_planning(True)
        self.motion_planning_interface.change_body_planning(True)
        self.motion_planning_interface.generate_motion_command()

        self.ref_torques, self.ref_tau_ff, self.ref_dof_pos, self.ref_dof_vel = self.mit_controller.step_run(
            self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos,
            self.dof_vel, self.feet_contact_state, self.motion_planning_interface.get_motion_command())
        self.motion_planning_interface.change_gait_planning(False)
        self.motion_planning_interface.change_body_planning(False)

    def calculate_vel_horizon_frame(self, base_lin_vel_measured, base_ang_vel_measured, base_lin_vel_ref, base_ang_vel_ref, base_quat_horizon, vel_weight=0.5):
        vel_weight = max(0., min(vel_weight, 1.))
        lin_vel = base_lin_vel_measured * vel_weight + base_lin_vel_ref * (1. - vel_weight)
        ang_vel = base_ang_vel_measured * vel_weight + base_ang_vel_ref * (1. - vel_weight)

        num_feet = len(self.feet_indices)
        tmp_vel = lin_vel.repeat(1, num_feet).view(self.num_envs, num_feet, 3)
        tmp_vel[..., 1] += ang_vel[:, 2].unsqueeze(-1) * self.body_half_length * self.side_coef
        # tmp_vel_horizon_frame = torch.zeros_like(tmp_vel)
        for i in range(num_feet):
            self.ref_lin_vel_horizon_feet[:, i, :] = quat_rotate(base_quat_horizon, tmp_vel[:, i, :])
        # ref_foothold_xy = tmp_vel_horizon_frame[..., :2] * stance_time.unsqueeze(-1).unsqueeze(-1) * 0.5

        return self.ref_lin_vel_horizon_feet

    def calculate_ref_foot_xy(self, phase_normed, vxy, period, duty):
        index_stance = torch.where(phase_normed < 0.5)
        index_swing = torch.where(phase_normed >= 0.5)
        phase = torch.zeros_like(phase_normed)
        phase[index_stance] = phase_normed[index_stance] / 0.5
        phase[index_swing] = (phase_normed[index_swing] - 0.5) / 0.5
        phase_ext = phase.unsqueeze(-1)
        period_ext = period.unsqueeze(-1).unsqueeze(-1)
        duty_ext = duty.unsqueeze(-1).unsqueeze(-1)

        self.ref_foot_pos_xy_horizon[index_stance] = ((0.5 - phase_ext) * vxy * (period_ext * duty_ext))[index_stance]
        # aaa = ((0.5 - phase_ext) * vxy * (period_ext * duty_ext))[index_stance]
        self.ref_foot_vel_xy_horizon[index_stance] = -vxy[index_stance]
        self.ref_foot_pos_xy_horizon[index_swing] = ((6 * torch.pow(phase_ext, 5) - 15 * torch.pow(phase_ext, 4) + 10 * torch.pow(phase_ext, 3) - (1. - duty_ext) * phase_ext - 0.5 * duty_ext) * vxy * period_ext)[index_swing]
        self.ref_foot_vel_xy_horizon[index_swing] = ((30 * torch.pow(phase_ext, 4) - 60 * torch.pow(phase_ext, 3) + 30 * torch.pow(phase_ext, 2)) * vxy / (1. - duty_ext) - vxy)[index_swing]

    def _cal_pd(self, tau_ff_mpc, q_des, qd_des, kp: Union[torch.Tensor, float] = 20., kd: Union[torch.Tensor, float] = 0.5):
        v_max = 20.0233
        v_max /= 1.0
        tau_max = 33.5 * 1.0
        k = -3.953886
        torques = torch.clip(tau_ff_mpc + kp * (q_des - self.dof_pos) + kd * (qd_des - self.dof_vel), -tau_max, tau_max)
        tmp_max_torque = torch.clip(k * (self.dof_vel - v_max), 0, tau_max)
        tmp_min_torque = torch.clip(k * (self.dof_vel + v_max), -tau_max, 0)
        torques[:] = torch.where(self.dof_vel > tau_max / k + v_max,
                                 torch.clip(torques, -tau_max * torch.ones_like(torques), tmp_max_torque), torques)
        torques[:] = torch.where(self.dof_vel < -(tau_max / k + v_max),
                                 torch.clip(torques, tmp_min_torque, tau_max * torch.ones_like(torques)), torques)
        return torques

    def calculate_C_des(self, phase_norm):
        self.ref_phase_C_des[:] = self.ref_phase_trans_distribution.cdf(phase_norm) * (1 - self.ref_phase_trans_distribution.cdf(phase_norm - 0.5)) + self.ref_phase_trans_distribution.cdf(phase_norm - 1)
        # self.ref_phase_C_des[:] = torch.where(phase_norm < 0.5, 1.0, 0.0)
        # print(f"ref_phase_C_des: {self.ref_phase_C_des[0]}")

    def calculate_foot_pos_track_weight(self, phase_norm):
        self.foot_pos_track_weight[:] = gaussian(phase_norm) + gaussian(phase_norm - 0.5) + gaussian(phase_norm - 1)
        # self.foot_pos_track_weight[:] = 1.0

    def calculate_Rw_matrix(self, pitch_yaw: torch.Tensor, R: torch.Tensor):
        pitch = pitch_yaw[:, 0]
        yaw = pitch_yaw[:, 1]

        cp = torch.cos(pitch)
        sp = torch.sin(pitch)
        cy = torch.cos(yaw)
        sy = torch.sin(yaw)

        # Add a small epsilon to cp to avoid division by zero
        epsilon = 1e-6
        cp = torch.where(torch.abs(cp) < epsilon, epsilon * torch.sign(cp), cp)

        R[:, 0, 0] = cy / cp
        R[:, 0, 1] = sy / cp
        R[:, 0, 2] = 0
        R[:, 1, 0] = -sy
        R[:, 1, 1] = cy
        R[:, 1, 2] = 0
        R[:, 2, 0] = cy * sp / cp
        R[:, 2, 1] = sy * sp / cp
        R[:, 2, 2] = 1

    def run_test(self):
        while True:
            if self.common_step_counter % 2 == 0:
                self.render()

            vel_commands = self.commands[:, :3].clone()
            vel_commands[:, 0] = 1.
            vel_commands[:, 1] = 0.
            vel_commands[:, 2] = 0.

            gait_period_offset = (self.gait_periods - 0.5).unsqueeze(-1)
            gait_duty_cycle_offset = (self.gait_commands[:, 1] - 0.5).unsqueeze(-1)
            gait_phase_offset = self.gait_commands[:, 2:6].clone()
            gait_phase_offset[:, 3] = self.ref_phase_current[:, 1]
            # gait_phase_offset[:, 3] = self.calculate_phase(self.ref_phase_sincos_current[:, 2],
            #                                                self.ref_phase_sincos_current[:, 3])
            body_height_offset = self.height_commands - 0.3
            gait_phase_offset[:, :2] = gait_phase_offset[:, :2].clone() - 0.5

            self.motion_planning_interface.update_gait_planning(True, gait_period_offset, gait_duty_cycle_offset,
                                                                gait_phase_offset, None)
            self.motion_planning_interface.update_body_planning(True, None, None, None, None,
                                                                vel_commands)
            self.motion_planning_interface.generate_motion_command()
            # ------------------------------------------------------------------------------------------------------------

            torques_est, tau_ff_mpc, q_des, qd_des = self.mit_controller.step_run(self.controller_reset_buf,
                                                                                  self.base_quat,
                                                                                  self.base_ang_vel, self.base_lin_acc,
                                                                                  self.dof_pos, self.dof_vel,
                                                                                  self.feet_contact_state,
                                                                                  self.motion_planning_interface.get_motion_command())
            self.motion_planning_interface.change_gait_planning(False)
            self.motion_planning_interface.change_body_planning(False)

            for i in range(self.decimation):
                # torques, _, _, _ = self._cal_torque()
                torques = self._cal_pd(tau_ff_mpc, q_des, qd_des, kp=self.Kp, kd=self.Kd)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
                self.torques[:] = torques.view(self.torques.shape)
                self.gym.simulate(self.sim)
                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
                self.update_pre_state()

                if i < self.decimation - 1:
                    _, _, _, _ = self.mit_controller.step_run(self.controller_reset_buf,
                                                              self.base_quat,
                                                              self.base_ang_vel, self.base_lin_acc,
                                                              self.dof_pos, self.dof_vel,
                                                              self.feet_contact_state,
                                                              self.motion_planning_interface.get_motion_command())
                    self.motion_planning_interface.change_gait_planning(False)
                    self.motion_planning_interface.change_body_planning(False)

            self.progress_buf += 1
            self.randomize_buf += 1
            self.common_step_counter += 1
            self.gait_commands_count += 1
            if self.push_flag and self.common_step_counter % 300 == 0:  ### wsh_annotation: self.push_interval > 0
                self.push_robots()
            self.check_termination()
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

# terrain generator
from isaacgym.terrain_utils import *
from isaacgymenvs.utils.custom_terrain import *
from multiprocessing import shared_memory, Process

class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 30
        self.num_per_env = 2
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [np.sum(cfg["terrainProportions"][:i + 1]) for i in range(len(cfg["terrainProportions"]))]
        self.add_terrain_obs = cfg["addTerrainObservation"]
        self.use_multiprocess = cfg["useMultiprocess"]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.num_per_env = int(num_robots / self.num_maps)
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))
        self.env_target_points = np.zeros((self.env_rows, self.env_cols, 8, 3))
        self.env_target_points_relative = np.zeros((8, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        if self.use_multiprocess:
            self.shm_height_field_raw = shared_memory.SharedMemory(create=True, size=self.height_field_raw.nbytes)
            self.shm_env_origins = shared_memory.SharedMemory(create=True, size=self.env_origins.nbytes)
            self.tmp_height_field_raw = np.ndarray(shape=self.height_field_raw.shape, dtype=self.height_field_raw.dtype,
                                                   buffer=self.shm_height_field_raw.buf)
            self.tmp_env_origins = np.ndarray(shape=self.env_origins.shape, dtype=self.env_origins.dtype,
                                              buffer=self.shm_env_origins.buf)
            self.tmp_height_field_raw[:] = self.height_field_raw[:]
            self.tmp_env_origins[:] = self.env_origins[:]

        t1 = time.time()

        if cfg["curriculum"]:
            self.curriculum(num_robots, num_terrains=self.env_cols, num_levels=self.env_rows)
        else:
            self.randomized_terrain(num_terrains=self.env_cols, num_levels=self.env_rows)

        t2 = time.time()
        print(f"time1: {(t2 - t1)} s")

        if self.use_multiprocess:
            self.height_field_raw[:] = self.tmp_height_field_raw[:]
            self.env_origins[:] = self.tmp_env_origins[:]

        self.heightsamples = self.height_field_raw

        t1 = time.time()

        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale,
                                                                       self.vertical_scale, cfg["slopeTreshold"])

        t2 = time.time()
        print(f"time2: {(t2 - t1)} s")

        # get target points
        rel_x = 0.5 * self.env_length
        rel_y = 0.5 * self.env_width
        bias = 0.1 * self.env_width
        self.env_target_points_relative[0] = np.array([rel_x, bias, 0.0])
        self.env_target_points_relative[1] = np.array([rel_x, rel_y, 0.0])
        self.env_target_points_relative[2] = np.array([rel_x, -rel_y, 0.0])
        self.env_target_points_relative[3] = np.array([-rel_x, -bias, 0.0])
        self.env_target_points_relative[4] = np.array([-rel_x, rel_y, 0.0])
        self.env_target_points_relative[5] = np.array([-rel_x, -rel_y, 0.0])
        self.env_target_points_relative[6] = np.array([bias, rel_y, 0.0])
        self.env_target_points_relative[7] = np.array([-bias, -rel_y, 0.0])
        for i in range(self.env_rows):
            for j in range(self.env_cols):
                self.env_target_points[i, j] = self.env_origins[i, j] + self.env_target_points_relative


        if self.use_multiprocess:
            self.shm_height_field_raw.close()
            self.shm_env_origins.close()
            self.shm_height_field_raw.unlink()
            self.shm_env_origins.unlink()

    def randomized_terrain(self, num_terrains, num_levels):
        if not self.use_multiprocess:
            for j in range(num_terrains):
                self._randomized_terrain_index(j, num_levels, self.height_field_raw, self.env_origins)
        else:
            ps = []
            for j in range(num_terrains):
                ps.append(Process(target=self._randomized_terrain_index, args=(j, num_levels, self.tmp_height_field_raw, self.tmp_env_origins)))
            for p in ps:
                p.start()
            for p in ps:
                p.join()

    def curriculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0

        if not self.use_multiprocess:
            for j in range(num_terrains):
                self._curriculum_index(j, num_robots, num_terrains, num_levels, self.height_field_raw, self.env_origins)
        else:
            ps = []
            for j in range(num_terrains):
                ps.append(Process(target=self._curriculum_index, args=(j, num_robots, num_terrains, num_levels, self.tmp_height_field_raw, self.tmp_env_origins)))
            for p in ps:
                p.start()
            for p in ps:
                p.join()

    def _randomized_terrain_index(self, j, num_levels, height_field_raw, env_origins):
        # # Env coordinates in the world
        # (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

        for i in range(num_levels):
            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                                 width=self.width_per_env_pixels,
                                 length=self.length_per_env_pixels,
                                 ### wsh_annotation: modify 'width_per_env_pixels' to 'length_per_env_pixels'
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0, 1)
            uniform_difficulty = np.random.uniform(0.5, 1)
            # uniform_difficulty = 1
            if not self.add_terrain_obs or True:
                if choice < -0.15:
                    pass
                elif choice < -0.35:
                    random_uniform_terrain(terrain, min_height=-0.02 * uniform_difficulty,
                                           max_height=0.02 * uniform_difficulty, step=0.05, downsampled_scale=0.5)
                # elif choice < 0.6:
                #     slope = 0.6 * difficulty
                #     if choice < 0.5:
                #         slope *= -1
                #     pyramid_sloped_terrain(terrain, slope=slope, platform_size=0.5)
                #     if np.random.choice([0, 1]):
                #         random_uniform_terrain(terrain, min_height=-0.005 * uniform_difficulty,
                #                                max_height=0.005 * uniform_difficulty, step=0.05, downsampled_scale=0.5)
                # elif choice < 0.8:
                #     step_height = 0.15 * difficulty
                #     if choice < 0.7:
                #         step_height *= -1
                #     pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                #     if np.random.choice([0, 1]):
                #         random_uniform_terrain(terrain, min_height=-0.005 * uniform_difficulty,
                #                                max_height=0.005 * uniform_difficulty, step=0.05, downsampled_scale=0.5)
                else:
                    # custom_up_step_terrain(terrain, 2, (i + 1)*0.005 + 0.05)
                    custom_up_step_terrain(terrain, 2, 0.1)
                    # max_height = 0.05 * difficulty
                    # discrete_obstacles_terrain(terrain, max_height=max_height, min_size=1., max_size=2., num_rects=200,
                    #                            platform_size=3.)
                    # if np.random.choice([0, 1]):
                    #     random_uniform_terrain(terrain, min_height=-0.005 * uniform_difficulty,
                    #                            max_height=0.005 * uniform_difficulty, step=0.05, downsampled_scale=0.5)

            else:
                if True:
                    pass
                elif choice < 1.1:
                    if np.random.choice([0, 1]):
                        pyramid_sloped_terrain(terrain, np.random.choice(
                            [-0.3, -0.2, 0, 0.2, 0.3]))  ### wsh_annotation: slope_angle = atan(slope)
                        # random_uniform_terrain(terrain, min_height=-0.03, max_height=0.03, step=0.03,
                        #                        downsampled_scale=0.2)
                    else:
                        pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                elif choice < 0.6:
                    # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                    step_height = np.random.choice([-0.15, 0.15])
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < 1.:
                    discrete_obstacles_terrain(terrain, 0.15, 1., 2., 40, platform_size=3.)

            height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def _curriculum_index(self, j, num_robots, num_terrains, num_levels, height_field_raw, env_origins):
        for i in range(num_levels):
            terrain = SubTerrain("terrain",
                                 width=self.width_per_env_pixels,
                                 length=self.length_per_env_pixels,
                                 ### wsh_annotation: modify 'width_per_env_pixels' to 'length_per_env_pixels'
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)
            difficulty = (i + 1) / num_levels  # - 1
            choice = j / num_terrains

            if not self.add_terrain_obs or True:
                if choice < 0.1:
                    pass
                elif choice < 0.2:
                    random_uniform_terrain(terrain, min_height=-0.1 * difficulty,
                                           max_height=0.1 * difficulty, step=0.05, downsampled_scale=0.5)
                elif choice < 0.4:
                    slope = 0.8 * difficulty
                    uniform_height = 0.003 * difficulty
                    if choice < 0.3:
                        slope *= -1
                        pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                        if choice >= 0.25:
                            random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height,
                                                   step=0.05, downsampled_scale=0.5)
                    else:
                        pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                        if choice >= 0.35:
                            random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height,
                                                   step=0.05, downsampled_scale=0.5)
                elif choice < 0.7:
                    step_height = 0.15 * difficulty
                    uniform_height = 0.02 * difficulty
                    if choice < 0.55:
                        step_height *= -1
                        pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                        if choice >= 0.5:
                            random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height,
                                                   step=0.05, downsampled_scale=0.5)
                    else:
                        pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                        if choice >= 0.65:
                            random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height,
                                                   step=0.05, downsampled_scale=0.5)
                else:
                    max_height = 0.1 * difficulty
                    uniform_height = 0.005 * difficulty
                    discrete_obstacles_terrain(terrain, max_height=max_height, min_size=1., max_size=2.,
                                               num_rects=200, platform_size=3.)
                    if choice >= 0.85:
                        random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height, step=0.05,
                                               downsampled_scale=0.5)
            else:
                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.02*difficulty, max_height=0.02*difficulty, step=0.02, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0.,
                                            platform_size=3.)

            # Heightfield coordinate system
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels
            height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            # robots_in_map = num_robots_per_map
            # if j < left_over:
            #     robots_in_map += 1

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
    def _curriculum_index2(self, j, num_robots, num_terrains, num_levels, height_field_raw, env_origins):
        for i in range(num_levels):
            terrain = SubTerrain("terrain",
                                 width=self.width_per_env_pixels,
                                 length=self.length_per_env_pixels,
                                 ### wsh_annotation: modify 'width_per_env_pixels' to 'length_per_env_pixels'
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)
            difficulty = (i + 1) / num_levels  # - 1
            choice = j / num_terrains

            if not self.add_terrain_obs:
                if choice < 0.2:
                    pass
                elif choice < 0.4:
                    random_uniform_terrain(terrain, min_height=-0.05 * difficulty,
                                           max_height=0.05 * difficulty, step=0.05, downsampled_scale=0.5)
                elif choice < 0.6:
                    slope = 0.2 * difficulty
                    uniform_height = 0.003 * difficulty
                    if choice < 0.5:
                        slope *= -1
                        pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                        if choice >= 0.45:
                            random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height,
                                                   step=0.05, downsampled_scale=0.5)
                    else:
                        pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                        if choice >= 0.55:
                            random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height,
                                                   step=0.05, downsampled_scale=0.5)
                elif choice < 0.8:
                    step_height = 0.05 * difficulty
                    uniform_height = 0.02 * difficulty
                    if choice < 0.7:
                        step_height *= -1
                        pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                        if choice >= 0.65:
                            random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height,
                                                   step=0.05, downsampled_scale=0.5)
                    else:
                        pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                        if choice >= 0.75:
                            random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height,
                                                   step=0.05, downsampled_scale=0.5)
                else:
                    max_height = 0.03 * difficulty
                    uniform_height = 0.005 * difficulty
                    discrete_obstacles_terrain(terrain, max_height=max_height, min_size=1., max_size=2.,
                                               num_rects=200, platform_size=3.)
                    if choice >= 0.9:
                        random_uniform_terrain(terrain, min_height=-uniform_height, max_height=uniform_height, step=0.05,
                                               downsampled_scale=0.5)
            else:
                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                    random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.025, downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1., 2., 40, platform_size=3.)
                else:
                    stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=0.1, max_height=0.,
                                            platform_size=3.)

            # Heightfield coordinate system
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels
            height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

            # robots_in_map = num_robots_per_map
            # if j < left_over:
            #     robots_in_map += 1

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
#
#
@torch.jit.script
def vec_rotate_z(vec_xy, angle):
    shape = vec_xy.shape
    vec_xy = vec_xy.reshape(-1, 2)
    angle = angle.reshape(-1)
    quat = torch.zeros(shape[0], 4, device=vec_xy.device)
    quat[:, 2] = torch.sin(angle / 2)
    quat[:, 3] = torch.cos(angle / 2)
    vec = torch.stack([vec_xy[:, 0], vec_xy[:, 1], torch.zeros_like(vec_xy[:, 0])], dim=1)
    vec_rotated = quat_apply(quat, vec)
    return vec_rotated[:, :2].reshape(shape)

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def get_euler_xyz2(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=1)

def gaussian(x, mu=0, sigma=0.1):
    return torch.exp(-(x - mu)**2 / (2 * sigma**2))


def generate_filename(prefix):
    import datetime
    # 获取当前时间
    now = datetime.datetime.now()
    # 将时间格式化为字符串，例如：2022-01-01-12-30-00
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    # 将前缀和时间戳拼接成文件名
    filename = f"{prefix}_{timestamp}.csv"
    return filename

import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from isaacgymenvs.utils.reformat import omegaconf_to_dict
if __name__ == "__main__":
    cfg = None
    with initialize(config_path="../cfg"):
        cfg = compose(config_name="config", overrides=[f"task=A1Dynamics"])
        cfg_dict = omegaconf_to_dict(cfg.task)
        cfg_dict['env']['numEnvs'] = 6
    # cfg = cfg_dict
    rl_device = 'cuda:0'
    sim_device = 'cuda:0'
    graphics_device_id = 0
    headless = False
    virtual_screen_capture = False
    force_render = True
    a1 = A1Record(cfg_dict, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)
    a1.run_test()
