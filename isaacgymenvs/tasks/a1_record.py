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
from .base.vec_task import VecTask

import torch
from typing import Tuple, Dict

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, normalize, quat_rotate, quat_apply, quat_rotate_inverse, copysign

from isaacgymenvs.utils.circle_buffer import CircleBuffer

from isaacgymenvs.utils.observation_utils import ObservationBuffer

from isaacgymenvs.utils.controller_bridge import SingleControllerBridge, VecControllerBridge

from isaacgymenvs.utils.motion_planning_interface import MotionPlanningInterface

from isaacgymenvs.utils.gait_tracking_policy import GaitTrackingPolicy

import random

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
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
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
        self.dt = self.decimation * self.cfg["sim"]["dt"]
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

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

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
        self.feet_lin_vel_body = torch.zeros_like(self.feet_lin_vel_world)
        self.feet_position_hip = torch.zeros_like(self.feet_position_body)
        hip_position_rel_body = self.cfg["env"]["urdfAsset"]["hip_position_rel_body"]
        self.hip_position_rel_body = torch.tensor(hip_position_rel_body, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_position_hip[:] = self.feet_position_body - self.hip_position_rel_body

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
        self.command_lin_vel_x = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,
                                    requires_grad=False)
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

        self.ref_phase_sincos_current = torch.zeros(self.num_envs, 8, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_phase_pi_current = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.ref_delta_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.feet_phase_sincos = torch.zeros_like(self.ref_phase_sincos_current)

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

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.last_actions_raw = torch.zeros_like(self.actions)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.feet_contact_state = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) ### wsh_annotation: 1->contact
        self.feet_contact_state_obs = torch.zeros_like(self.feet_contact_state)

        self.height_points = self.init_height_points()
        self.measured_heights = None
        # joint positions offsets
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device,
                                                requires_grad=False)


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
                             "gait_phase_shape": torch_zeros()}

        self.base_quat = self.root_states[:, 3:7]
        self.euler_xyz = get_euler_xyz2(self.base_quat)
        self.euler_roll = self.euler_xyz.view(self.num_envs, 1, 3)[..., 0]
        self.euler_pitch = self.euler_xyz.view(self.num_envs, 1, 3)[..., 1]
        self.euler_yaw = self.euler_xyz.view(self.num_envs, 1, 3)[..., 2]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # acceleration
        self.last_base_lin_vel_rel_world = self.root_states[:, 7:10].clone().detach()
        self.gravity_acc = torch.tensor([0., 0., -9.81], dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_acc = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                        requires_grad=False) - self.gravity_acc

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
                                  "feetPositionRelHip": self.feet_position_hip,
                                  "feetLinVelRelHip": self.feet_lin_vel_body,
                                  "gaitParamsAct": self.gait_params_act,
                                  "armature_coeffs_real": self.armature_coeffs_real,
                                  "friction_coeffs_real": self.friction_coeffs_real,
                                  "power_norm": self.power_norm,
                                  "command_lin_vel_x": self.command_lin_vel_x,
                                  "vx_mean": self.vx_mean,
                                  "feet_phase_sincos": self.feet_phase_sincos}

        self.obs_combination = self.cfg["env"]["learn"]["observationConfig"]["combination"]
        for key in self.obs_combination.keys():
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
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.target_points_list = torch.from_numpy(self.terrain.env_target_points).to(self.device).to(torch.float)
            self.target_points = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
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
                rigid_shape_prop[s].friction = 1.0
            #     # rigid_shape_prop[s].restitution = restitution_buckets[i % num_buckets]
            #     rigid_shape_prop[s].friction = 1.
            #     rigid_shape_prop[s].rolling_friction = 0.
            #     rigid_shape_prop[s].torsion_friction = 0.
            #     rigid_shape_prop[s].restitution = self.cfg["env"]["terrain"]["restitution"]
            #     rigid_shape_prop[s].contact_offset = 0.001
            #     rigid_shape_prop[s].rest_offset = -0.003

            for j in range(self.num_dof):
                dof_props['driveMode'][j] = gymapi.DOF_MODE_EFFORT  # gymapi.DOF_MODE_POS
                # dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
                # dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd
                dof_props['armature'][j] = self.armature_coeffs[i]  # 0.01
                dof_props['armature'][j] = 0.01

            self.gym.set_asset_rigid_shape_properties(a1_asset, rigid_shape_prop)
            a1_handle = self.gym.create_actor(env_handle, a1_asset, start_pose, "a1", i, 1, 0)
            self.gym.set_actor_dof_properties(env_handle, a1_handle, dof_props)

            rigid_body_prop = self.gym.get_actor_rigid_body_properties(env_handle, a1_handle)

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
        self.modify_vel_command()
        # self.commands[:, 0] = 1.0

        # self.modify_desired_gait_command()
        # self.calculate_ref_timing_phase()

        # self.modify_desired_height_command()

        self.record_commands()

        if self.add_terrain_obs:
            self.measured_heights = self.get_heights()
            self.terrain_height[:] = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)
            self.obs_buffer_dict["heightMeasurement"].record(self.terrain_height)

        self.obs_buf[:] = torch.cat([self.obs_buffer_dict[key].get_index_data(self.obs_combination[key]) for key in self.obs_combination.keys()], dim=-1)

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
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]
        # rew_lin_vel_z = self.base_lin_vel_error_square_accumulated[:, 2] / self.decimation * self.rew_scales["lin_vel_z"]
        # rew_ang_vel_xy = torch.sum(self.base_ang_vel_error_square_accumulated[:, :2], dim=1) / self.decimation * self.rew_scales["ang_vel_xy"]
        # self.base_lin_vel_error_square_accumulated[:] = 0
        # self.base_ang_vel_error_square_accumulated[:] = 0

        # orientation penalty TODO relating to velocity
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        # base height penalty
        rew_base_height = torch.square(self.root_states[:, 2] - self.height_commands.squeeze()) * self.rew_scales["base_height"]  # TODO(completed) add target base height to cfg
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
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"]
        rew_stumble = torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
                                4 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1).float() * self.rew_scales["stumble"]

        # action rate penalty TODO action is uncertain
        rew_action_rate = torch.norm(self.last_actions - self.actions, dim=1) * self.rew_scales["action_rate"]
        # rew_action_rate = torch.norm(self.last_actions[:, :12] - self.actions[:, :12], dim=1) * self.rew_scales["action_rate"]
        # rew_action_rate += torch.norm(self.last_actions[:, 12:] - self.actions[:, 12:], dim=1) * (-1.0e-3)*self.dt
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

        buf_length = 5
        body_height_buf = self.obs_buffer_dict["bodyPos"].get_len_data_raw(buf_length)[:, 2, :]
        lin_vel_buf = self.obs_buffer_dict["linearVelocity"].get_len_data_raw(buf_length)
        ang_vel_buf = self.obs_buffer_dict["angularVelocity"].get_len_data_raw(buf_length)
        projectedGravity_buf = self.obs_buffer_dict["projectedGravity"].get_len_data_raw(buf_length)
        motor_vel_buf = self.obs_buffer_dict["dofVelocity"].get_len_data_raw(buf_length)
        motor_torque_buf = self.obs_buffer_dict["motorTorque"].get_len_data_raw(buf_length)
        motor_torque_buf2 = self.obs_buffer_dict["motorTorque"].get_len_data_raw(buf_length)
        feet_force_buf = self.obs_buffer_dict["feetForce"].get_len_data_raw(buf_length)
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
        roll_sin_mean = torch.mean(roll_sin_buf, dim=-1)
        pitch_sin_mean = torch.mean(pitch_sin_buf, dim=-1)
        power_mean_each = torch.mean(power, dim=-1)
        power_mean_total = torch.sum(power_mean_each, dim=-1)
        power_max_mean_each = torch.max(power_mean_each, dim=-1).values
        power_max_mean_std = torch.std(power_mean_each[:, [1, 2, 4, 5, 7, 8, 10, 11]], dim=-1)
        torque_max_each = torch.max(torch.abs(motor_torque_buf2), dim=-1).values
        torque_max_each_mean = torch.mean(torque_max_each, dim=-1)
        torque_max_each_std = torch.std(torque_max_each[:, [1, 2, 4, 5, 7, 8, 10, 11]], dim=-1)

        tmp_vel_mean = torch.stack([v_x_mean, v_y_mean, v_z_mean, v_roll_mean, v_pitch_mean, v_yaw_mean], dim=1)
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


        # total reward
        self.rew_buf[:] = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height + \
                          rew_torques + rew_joint_acc + rew_knee_collision + rew_action_rate + rew_air_time + rew_hip + rew_stumble + rew_energy + rew_power + rew_survival + \
                          rew_power_max + rew_power_max_std + rew_feet_force_max + rew_feet_force_max_std + rew_torque_max + rew_torque_max_std + rew_fallen_over + rew_delta_torques + rew_dof_bias + \
                          rew_gait_tracking + rew_gait_trans_rate + rew_gait_phase_timing + rew_gait_phase_shape
        # self.rew_buf[:] = rew_orient
        # self.rew_buf[:] = rew_lin_vel_xy + rew_ang_vel_z + rew_energy + rew_survival
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

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        positions_offset = 1.0
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

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1],
                                                     (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1],
                                                     (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1],
        #                                              (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 2] = self.commands[env_ids, 3]
        # self.commands[env_ids, 1] = 0.
        # self.commands[env_ids, 2] = 0.
        # self.commands[env_ids, 0] = 1.0
        # self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.xy_velocity_threshold).unsqueeze(1)  # set small commands to zero. wsh_annotation: TODO 0.25 ?
        # self.modify_vel_command()
        self.commands[env_ids] *= ~((self.commands[env_ids, :3].abs() < self.xy_velocity_threshold).all(dim=-1)).unsqueeze(1)
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
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        # b = torch.tensor([0, 1], dtype=torch.long, device=self.device, requires_grad=False)
        # a = self.rigid_body_states_reshape[b][:, self.feet_indices, :]
        # aa = a[..., 0:3]
        # aaa = aa.view(len(env_ids), -1)
        self.feet_position_world[env_ids] = self.rigid_body_states_reshape[env_ids][:, self.feet_indices, 0:3].reshape(len(env_ids), -1)
        self.feet_lin_vel_world[env_ids] = self.rigid_body_states_reshape[env_ids][:, self.feet_indices, 7:10].reshape(len(env_ids), -1)
        for i in range(len(self.feet_indices)):
            self.feet_position_body[env_ids, i * 3: i * 3 + 3] = quat_rotate_inverse(self.base_quat[env_ids], self.feet_position_world[env_ids, i * 3: i * 3 + 3] - self.root_states[env_ids, 0:3])
            self.feet_lin_vel_body[env_ids, i * 3: i * 3 + 3] = quat_rotate_inverse(self.base_quat[env_ids],self.feet_lin_vel_world[env_ids, i * 3: i * 3 + 3] - self.root_states[env_ids, 7:10])
        self.feet_position_hip[env_ids] = self.feet_position_body[env_ids] - self.hip_position_rel_body

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
        self.feet_force[env_ids] = self.contact_forces[env_ids][:, self.feet_indices].reshape(len(env_ids), -1)
        self.feet_force[env_ids] = 0.0

        self.controller_reset_buf[env_ids] = 1

        self.height_commands[env_ids] = 0.3

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
                self.episode_sums[key][env_ids] / (self.progress_buf[env_ids] * self.dt + 1.e-5))  # / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
            # print(self.extras["episode"]['rew_' + key])
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

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
        # self.root_states[env_ids, 7:9] = torch.tensor([0., 1.0], device=self.device)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # if len(env_ids):
        #     if env_ids[0] == 0:
        #         print("push-push-push-push-push-push-push-push-push-push-push")

    def pre_physics_step(self, actions):
        # t = time.time()

        self.actions[:] = actions.clone().to(self.device)
        # print(f"action_raw_0: {self.actions[0]}")
        self.last_actions_raw[:] = self.actions.clone()
        self.actions[:] = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.actions[:] *= self.action_scale

        dof_pos_desired = self.default_dof_pos
        joint_vel_des = 0.
        tau_ff = 0.
        gait_period_offset = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) - 0.2
        gait_phase_offset = torch.zeros_like(gait_period_offset)
        gait_duty_cycle_offset = torch.zeros_like(gait_period_offset)

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
        # vel_commands = self.commands[:, :3].clone()
        # vel_commands[:, 0] += self.actions[:, 0].clone()  # vel_x offset
        # vel_commands[:, 2] += self.actions[:, 1].clone() * 2.0  # omega offset
        # self.body_orientation_commands[:, 0] = self.actions[:, 0].clone() * 0.2  # [-0.4, 0.4]
        # self.body_orientation_commands[:, 1] = self.actions[:, 1].clone() * 0.6  # [-0.8, 0.8]
        self.feet_mid_bias_xy[:, 0:2] = self.actions[:, 0].unsqueeze(-1).repeat(1, 2) * 0.1  # [-0.1, 0.1]
        self.feet_mid_bias_xy[:, 2:4] = self.actions[:, 1].unsqueeze(-1).repeat(1, 2) * 0.1
        self.feet_mid_bias_xy[:, [4, 6]] = self.actions[:, 2].unsqueeze(-1).repeat(1, 2) * 0.1  # [-0.1, 0.1]
        self.feet_mid_bias_xy[:, [5, 7]] = self.actions[:, 3].unsqueeze(-1).repeat(1, 2) * 0.1
        # self.feet_lift_height_bias[:, :4] = torch.exp(self.actions[:, 6:10].clone()) * 0.1 - 0.05  # [-0.15, 0.15]
        # self.feet_lift_height_bias[:, 4:8] = self.actions[:, 10:14].clone() * 0.4  # [-0.4, 0.4]
        # self.des_feet_pos_rel_hip[:] = self.actions[:, 18:30].clone() * 0.15  # [-0.15, 0.15]

        # self.feet_lift_height_bias[:, :4] = self.actions[:, 0:4].clone() * 0.1  # [-0.15, 0.15]
        # self.feet_lift_height_bias[:, 4:8] = self.actions[:, 4:8].clone() * 0.4  # [-0.4, 0.4]
        # self.des_feet_pos_rel_hip[:] = self.actions.clone()  # [-0.15, 0.15]

        # print(self.feet_mid_bias_xy[0])
        # print(self.feet_lift_height_bias[0])


        # ------------------------------------------- follow gait commands -------------------------------------------
        # gait_period_offset = (self.gait_periods - 0.5).unsqueeze(-1)
        # gait_duty_cycle_offset = (self.gait_commands[:, 1] - 0.5).unsqueeze(-1)
        # gait_phase_offset = self.gait_commands[:, 2:6].clone()
        # gait_phase_offset[:, 3] = self.calculate_phase(self.ref_phase_sincos_current[:, 0], self.ref_phase_sincos_current[:, 1])
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
        # body_height_offset = self.height_commands - 0.3
        # gait_phase_offset[:, :2] = gait_phase_offset[:, :2].clone() - 0.5
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
        self.motion_planning_interface.update_gait_planning(True, gait_period_offset, gait_duty_cycle_offset, gait_phase_offset, None)
        self.motion_planning_interface.update_body_planning(True, None, self.body_orientation_commands, None, None, self.commands[:, :3])
        self.motion_planning_interface.update_feet_mid_bias_xy(self.feet_mid_bias_xy)
        self.motion_planning_interface.update_feet_lift_height(self.feet_lift_height_bias)
        self.motion_planning_interface.update_des_feet_pos_rel_hip(self.des_feet_pos_rel_hip)
        self.motion_planning_interface.generate_motion_command()
        # ------------------------------------------------------------------------------------------------------------

        # dof_pos_desired = self.default_dof_pos

        kp = torch.zeros_like(self.dof_pos[0])
        kd = torch.zeros_like(self.dof_pos[0])
        kp[:] = 25
        kp[[0, 3, 6, 9]] = 5
        kd[:] = 1

        v_max = 20.0233
        v_max /= 1.0
        tau_max = 33.5 * 1.0
        k = -3.953886

        for i in range(self.decimation - 1):
            torques, tau_ff_mpc, q_des, qd_des = self.mit_controller.step_run(self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state, self.motion_planning_interface.get_motion_command())  # self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state
            self.motion_planning_interface.change_gait_planning(False)
            self.motion_planning_interface.change_body_planning(False)

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
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques[:] = torques.view(self.torques.shape)
            # self.torques = torques.view(self.torques.shape)
            if i % 10 == 0 and self.force_render:
                self.render()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            # self.gym.refresh_dof_state_tensor(self.sim)
            # self.record_state_test()

            self.update_pre_state()
            self.record_states_into_buffer()
            self.record_commands()

            # self.torques_square_accumulated += torch.square(torques)

        torques, tau_ff_mpc, q_des, qd_des = self.mit_controller.step_run(self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state, self.motion_planning_interface.get_motion_command())

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
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        self.torques[:] = torques.view(self.torques.shape)

        # self.record_state()
        # self.torques_square_accumulated += torch.square(torques)
        # self.record_state_test()

    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        self.gait_commands_count += 1
        self.schedule_random()
        # if self.push_flag and self.common_step_counter % self.push_interval == 0:  ### wsh_annotation: self.push_interval > 0
        #     self.push_robots()

        # # prepare quantities
        # self.base_quat[:] = self.root_states[:, 3:7]
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.update_pre_state()
        self.record_states_into_buffer()

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
            sphere_geom_target = gymutil.WireframeSphereGeometry(0.1, 32, 32, None, color=(0, 0, 1))

            if not self.add_terrain_obs:
                self.measured_heights = self.get_heights()

            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]),
                                               self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

                # draw target
                target_point = self.target_points[i]
                target_point_xy = target_point[:2] + self.terrain.border_size
                pts = (target_point_xy / self.terrain.horizontal_scale).long()
                target_point_z = self.height_samples[pts[0], pts[1]].cpu().item() * self.terrain.vertical_scale
                pose = gymapi.Transform(gymapi.Vec3(target_point[0], target_point[1], target_point_z), r=None)
                gymutil.draw_lines(sphere_geom_target, self.gym, self.viewer, self.envs[i], pose)

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
            record_dir = os.path.join('/home/wsh/Documents/pyProjects/IsaacGymEnvs/isaacgymenvs/runs', 'A1_2023-10-23_04-43-08(low_power_7_mpc)/record_test_data')
            os.makedirs(record_dir, exist_ok=True)
            # file_name = generate_filename('record_data_v-0_3_0')
            file_name = 'record_data-A1_2023-10-23_04-43-08(low_power_7_mpc)_v03_amu001_mpc_trot-05-075-05-05-00.csv'
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

    def update_pre_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.simulate_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.euler_xyz[:] = get_euler_xyz2(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.dof_pos_rel_init[:] = self.dof_pos - self.default_dof_pos

        self.feet_position_world[:] = self.rigid_body_states_reshape[:, self.feet_indices, 0:3].view(self.num_envs, -1)
        self.feet_lin_vel_world[:] = self.rigid_body_states_reshape[:, self.feet_indices, 7:10].view(self.num_envs, -1)
        # self.feet_position_body[:] = quat_rotate_inverse(self.base_quat, self.feet_position_world - self.root_states[:, 0:3])
        # self.feet_lin_vel_body[:] = quat_rotate_inverse(self.base_quat, self.feet_lin_vel_world - self.root_states[:, 7:10])
        for i in range(len(self.feet_indices)):
            self.feet_position_body[:, i * 3: i * 3 + 3] = quat_rotate_inverse(self.base_quat,
                                                                                     self.feet_position_world[:,
                                                                                     i * 3: i * 3 + 3] - self.root_states[
                                                                                                         :, 0:3])
            self.feet_lin_vel_body[:, i * 3: i * 3 + 3] = quat_rotate_inverse(self.base_quat,
                                                                                    self.feet_lin_vel_world[:,
                                                                                    i * 3: i * 3 + 3] - self.root_states[
                                                                                                        :, 7:10])
        self.feet_position_hip[:] = self.feet_position_body - self.hip_position_rel_body
        self.feet_force[:] = self.contact_forces[:, self.feet_indices].view(self.num_envs, -1)
        self.feet_contact_state[:] = self.contact_forces[:, self.feet_indices, 2] > self.stance_foot_force_threshold
        self.feet_contact_state_obs[:] = self.feet_contact_state.float() - 0.5

        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, ((self.root_states[:,
                                                                     7:10] - self.last_base_lin_vel_rel_world) / self.sim_params.dt - self.gravity_acc))
        self.last_base_lin_vel_rel_world[:] = self.root_states[:, 7:10].clone().detach()

        self.controller_reset_buf[:] = 0

        # self.base_lin_vel_error_square_accumulated += torch.square(self.tmp_lin_vel_command - self.base_lin_vel)
        # self.base_ang_vel_error_square_accumulated += torch.square(self.tmp_ang_vel_command - self.base_ang_vel)

        # self.record_states_into_buffer()

        # print("lin acc: ", self.base_lin_acc)
        # print("last vel:", self.last_base_lin_vel_rel_world)
        # print(f"body_height_real: {(self.root_states[0, 2] - 0.3) * 1000} mm")
        # print(f"base_lin_vel: {self.base_lin_vel[0]}")
        # print(f"joint_vel: {self.dof_vel[0]}")
        # print(f"torque: {self.torques[0]}")

    def modify_vel_command(self):
        env_ids = (self.progress_buf % self.commands_change_random_count == 0).nonzero(as_tuple=False).flatten()
        self.commands[env_ids, 0] = torch_rand_float(self.scheduled_command_x_range[0], self.scheduled_command_x_range[1],
                                                     (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.scheduled_command_y_range[0], self.scheduled_command_y_range[1],
                                                     (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(self.scheduled_command_yaw_range[0], self.scheduled_command_yaw_range[1],
                                                     (len(env_ids), 1), device=self.device).squeeze()
        # self.commands[env_ids, 2] = self.commands[env_ids, 3].clone()
        self.commands[:, 0] = 0.


        self.commands[:, 1] = 0.
        self.commands[:, 3] = 0.
        # if self.common_step_counter % 120 > 60:
        #     self.commands[:, 3] = 3
        # else:
        #     self.commands[:, 3] = 0

        # calculate omega command
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = self._heading_to_omega(heading)
        # self.commands[:, 2] = 5.
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

        # count = self.common_step_counter % 1000
        # if count < 700:
        #     n = int(self.common_step_counter / 100)
        #     self.commands[:, 0] = 0.3 * n
        # elif count < 800:
        #     self.commands[:, 0] = 1.0
        # elif count < 900:
        #     self.commands[:, 0] = 0.5
        # elif count < 1000:
        #     self.commands[:, 0] = 0.0

        # self.commands[:, 0] = 1.0
        # self.commands[:, 1] = 0.0
        # self.commands[:, 2] = 0.6

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
        # env_ids = (self.progress_buf % self.gait_commands_change_random_count == 0).nonzero(as_tuple=False).flatten()
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
        #
        # # self.gait_commands[env_ids, 2] = 0.5
        # # self.gait_commands[env_ids, 3] = 0.5
        # # self.gait_commands[env_ids, 4] = 0.0
        #
        # self.gait_periods[env_ids] = self.gait_commands[env_ids, 0]
        # self.ref_delta_phase[env_ids] = 2.0 * torch.pi * self.dt / self.gait_periods[env_ids]
        # self.gait_commands_count[env_ids] = 0

        # ----------------------------------- specific commands -----------------------------------
        if self.common_step_counter == 0:
            self.specific_gait_dict = {}
            self.specific_gait_dict["walk"] = torch.tensor([0.6, 0.7, 0.75, 0.5, 0.25, 0.0], device=self.device)
            self.specific_gait_dict["trot"] = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.0, 0.0], device=self.device)
            self.specific_gait_dict["flying_trot"] = torch.tensor([0.3, 0.25, 0.5, 0.5, 0.0, 0.0], device=self.device)
            self.specific_gait_dict["pace"] = torch.tensor([0.3, 0.7, 0.5, 0.0, 0.5, 0.0], device=self.device)
            self.specific_gait_dict["bound"] = torch.tensor([0.3, 0.3, 0.0, 0.5, 0.5, 0.0], device=self.device)
            self.specific_gait_dict["jump"] = torch.tensor([0.2, 0.5, 0.0, 0.0, 0.0, 0.0], device=self.device)
            self.specific_gait_dict["transverse_gallop"] = torch.tensor([0.24, 0.3, 0.25, 0.6, 0.85, 0.0], device=self.device)
            self.specific_gait_dict["random"] = torch.tensor([0.6, 0.6, 0.37, 0.32, 0.6, 0.0], device=self.device)
        if self.common_step_counter % 89 == 0:
            # tmp_gaits = list(self.specific_gait_dict.values())
            # index = torch.randint(0, len(tmp_gaits), (1,)).item()
            # self.gait_commands[:, :] = tmp_gaits[index]
            self.gait_commands[:, :] = self.specific_gait_dict["trot"]
            # self.gait_commands_count[:] = 0
            # print("to transverse_gallop")
        if self.common_step_counter % 179 == 0:
            self.gait_commands[:, :] = self.specific_gait_dict["trot"]
            # self.gait_commands_count[:] = 0
            # print("to flying_trot")
        self.gait_periods[:] = self.gait_commands[:, 0]
        self.ref_delta_phase[:] = 2.0 * torch.pi * self.dt / self.gait_periods

    def modify_desired_height_command(self):
        env_ids = (self.progress_buf % self.height_commands_change_random_count == 0).nonzero(as_tuple=False).flatten()
        self.height_commands[env_ids] = torch_rand_float(self.scheduled_command_height_range[0], self.scheduled_command_height_range[1],
                                                         (len(env_ids), 1), device=self.device)

        # specific commands
        self.height_commands[:] = 0.3

    def record_states_into_buffer(self):
        ### wsh_annotation: record new states into buffer
        for key in self.record_items:
            if key != "commands" and key != "heightMeasurement":
                self.obs_buffer_dict[key].record(self.obs_name_to_value[key])

    def record_commands(self):
        self.obs_buffer_dict["commands"].record(self.obs_name_to_value["commands"])

    def calculate_phase(self, sin_theta, cos_theta):
        # Calculate theta
        theta = torch.atan2(sin_theta.clone(), cos_theta.clone())
        # Adjust theta to the range [0, 2*pi]
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)
        phase = theta / (2 * torch.pi)
        return phase

    def schedule_random(self):
        if self.if_schedule_command:
            if self.common_step_counter == 24 * 0:
                self.scheduled_command_x_range[0] = -0.0
                self.scheduled_command_x_range[1] = 0.3
                self.scheduled_command_y_range[0] = -0.1
                self.scheduled_command_y_range[1] = 0.1
                self.scheduled_command_yaw_range[0] = -0.1
                self.scheduled_command_yaw_range[1] = 0.1
                self.scheduled_push_velocity_range[0] = -0.1
                self.scheduled_push_velocity_range[1] = 0.1
                self.scheduled_command_height_range[0] = 0.29
                self.scheduled_command_height_range[1] = 0.31
            elif self.common_step_counter == 24 * 200:
                self.scheduled_command_x_range[0] = -0.1
                self.scheduled_command_x_range[1] = 0.5
                self.scheduled_command_y_range[0] = -0.5
                self.scheduled_command_y_range[1] = 0.5
                self.scheduled_command_yaw_range[0] = -0.5
                self.scheduled_command_yaw_range[1] = 0.5
                self.scheduled_push_velocity_range[0] = -0.2
                self.scheduled_push_velocity_range[1] = 0.2
                self.scheduled_command_height_range[0] = 0.28
                self.scheduled_command_height_range[1] = 0.32
            elif self.common_step_counter == 24 * 400:
                self.scheduled_command_x_range[0] = -0.3
                self.scheduled_command_x_range[1] = 0.8
                self.scheduled_command_y_range[0] = -0.8
                self.scheduled_command_y_range[1] = 0.8
                self.scheduled_command_yaw_range[0] = -1.5
                self.scheduled_command_yaw_range[1] = 1.5
                self.scheduled_push_velocity_range[0] = -0.5
                self.scheduled_push_velocity_range[1] = 0.5
                self.scheduled_command_height_range[0] = 0.27
                self.scheduled_command_height_range[1] = 0.33
            elif self.common_step_counter == 24 * 600:
                self.scheduled_command_x_range[0] = -0.6
                self.scheduled_command_x_range[1] = 1.1
                self.scheduled_command_y_range[0] = -1.0
                self.scheduled_command_y_range[1] = 1.0
                self.scheduled_command_yaw_range[0] = -2.
                self.scheduled_command_yaw_range[1] = 2.
                self.scheduled_push_velocity_range[0] = -1.
                self.scheduled_push_velocity_range[1] = 1.
                self.scheduled_command_height_range[0] = 0.25
                self.scheduled_command_height_range[1] = 0.34
            elif self.common_step_counter == 24 * 800:
                self.scheduled_command_x_range[0] = -0.8
                self.scheduled_command_x_range[1] = 1.5
                self.scheduled_command_y_range[0] = -1.1
                self.scheduled_command_y_range[1] = 1.1
                self.scheduled_command_yaw_range[0] = -2.5
                self.scheduled_command_yaw_range[1] = 2.5
                self.scheduled_push_velocity_range[0] = -1.5
                self.scheduled_push_velocity_range[1] = 1.5
                self.scheduled_command_height_range[0] = 0.23
                self.scheduled_command_height_range[1] = 0.35
            elif self.common_step_counter == 24 * 1200:
                self.scheduled_command_x_range[0] = -1.0
                self.scheduled_command_x_range[1] = 2.0
                self.scheduled_command_y_range[0] = -1.2
                self.scheduled_command_y_range[1] = 1.2
                self.scheduled_command_yaw_range[0] = -3.0
                self.scheduled_command_yaw_range[1] = 3.0
                self.scheduled_push_velocity_range[0] = -2.
                self.scheduled_push_velocity_range[1] = 2.
                self.scheduled_command_height_range[0] = 0.20
                self.scheduled_command_height_range[1] = 0.36
            elif self.common_step_counter == 24 * 1600:
                self.scheduled_command_x_range[0] = -1.0
                self.scheduled_command_x_range[1] = 2.5
                self.scheduled_command_y_range[0] = -1.5
                self.scheduled_command_y_range[1] = 1.5
                self.scheduled_command_yaw_range[0] = -3.0
                self.scheduled_command_yaw_range[1] = 3.0
                self.scheduled_push_velocity_range[0] = -2.
                self.scheduled_push_velocity_range[1] = 2.
                self.scheduled_command_height_range[0] = 0.20
                self.scheduled_command_height_range[1] = 0.36
            elif self.common_step_counter == 24 * 2000:
                self.scheduled_command_x_range[0] = -1.0
                self.scheduled_command_x_range[1] = 3.0
                self.scheduled_command_y_range[0] = -1.5
                self.scheduled_command_y_range[1] = 1.5
                self.scheduled_command_yaw_range[0] = -3.14
                self.scheduled_command_yaw_range[1] = 3.14
                self.scheduled_push_velocity_range[0] = -2.
                self.scheduled_push_velocity_range[1] = 2.
                self.scheduled_command_height_range[0] = 0.20
                self.scheduled_command_height_range[1] = 0.36
            elif self.common_step_counter == 24 * 3000:
                self.scheduled_command_x_range[0] = -1.0
                self.scheduled_command_x_range[1] = 4.0
                self.scheduled_command_y_range[0] = -1.5
                self.scheduled_command_y_range[1] = 1.5
                self.scheduled_command_yaw_range[0] = -3.14
                self.scheduled_command_yaw_range[1] = 3.14
                self.scheduled_push_velocity_range[0] = -2.
                self.scheduled_push_velocity_range[1] = 2.
                self.scheduled_command_height_range[0] = 0.20
                self.scheduled_command_height_range[1] = 0.36
            elif self.common_step_counter == 24 * 4000:
                self.scheduled_command_x_range[0] = -1.0
                self.scheduled_command_x_range[1] = 5.0
                self.scheduled_command_y_range[0] = -1.5
                self.scheduled_command_y_range[1] = 1.5
                self.scheduled_command_yaw_range[0] = -3.14
                self.scheduled_command_yaw_range[1] = 3.14
                self.scheduled_push_velocity_range[0] = -2.
                self.scheduled_push_velocity_range[1] = 2.
                self.scheduled_command_height_range[0] = 0.20
                self.scheduled_command_height_range[1] = 0.36
            elif self.common_step_counter == 24 * 5000:
                self.scheduled_command_x_range[0] = -1.0
                self.scheduled_command_x_range[1] = 6.0
                self.scheduled_command_y_range[0] = -1.5
                self.scheduled_command_y_range[1] = 1.5
                self.scheduled_command_yaw_range[0] = -3.14
                self.scheduled_command_yaw_range[1] = 3.14
                self.scheduled_push_velocity_range[0] = -2.
                self.scheduled_push_velocity_range[1] = 2.
                self.scheduled_command_height_range[0] = 0.20
                self.scheduled_command_height_range[1] = 0.36

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

        self.ref_phase_pi_current[env_ids_init] = torch.cat((self.gait_commands[env_ids_init, 5].unsqueeze(-1), self.gait_commands[env_ids_init, 2:5]), dim=-1) * torch.pi * 2.0
        self.ref_phase_pi_current[env_ids_running] += self.ref_delta_phase[env_ids_running].unsqueeze(-1)
        self.ref_phase_pi_current[env_ids_running] %= 2 * torch.pi

        for i in range(4):
            self.ref_phase_sincos_current[:, 2 * i] = torch.sin(self.ref_phase_pi_current[:, i])
            self.ref_phase_sincos_current[:, 2 * i + 1] = torch.cos(self.ref_phase_pi_current[:, i])


# terrain generator
from isaacgym.terrain_utils import *
from multiprocessing import shared_memory, Process

class Terrain:
    def __init__(self, cfg, num_robots) -> None:
        self.type = cfg["terrainType"]
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1
        self.vertical_scale = 0.005
        self.border_size = 20
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
            uniform_difficulty = np.random.uniform(0, 1)
            # uniform_difficulty = 1
            if not self.add_terrain_obs:
                if choice < 0.0:
                    pass
                elif choice < 0.4:
                    random_uniform_terrain(terrain, min_height=-0.1 * uniform_difficulty,
                                           max_height=0.1 * uniform_difficulty, step=0.05, downsampled_scale=0.5)
                elif choice < 0.6:
                    slope = 0.6 * difficulty
                    if choice < 0.5:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=0.5)
                    if np.random.choice([0, 1]):
                        random_uniform_terrain(terrain, min_height=-0.005 * uniform_difficulty,
                                               max_height=0.005 * uniform_difficulty, step=0.05, downsampled_scale=0.5)
                elif choice < 0.8:
                    step_height = 0.15 * difficulty
                    if choice < 0.7:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                    if np.random.choice([0, 1]):
                        random_uniform_terrain(terrain, min_height=-0.005 * uniform_difficulty,
                                               max_height=0.005 * uniform_difficulty, step=0.05, downsampled_scale=0.5)
                else:
                    max_height = 0.2 * difficulty
                    discrete_obstacles_terrain(terrain, max_height=max_height, min_size=1., max_size=2., num_rects=200,
                                               platform_size=3.)
                    if np.random.choice([0, 1]):
                        random_uniform_terrain(terrain, min_height=-0.005 * uniform_difficulty,
                                               max_height=0.005 * uniform_difficulty, step=0.05, downsampled_scale=0.5)

            else:
                if choice < 1.1:
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
            difficulty = i / num_levels  # - 1
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
                    max_height = 0.1 + 0.3 * difficulty
                    uniform_height = 0.05 * difficulty
                    discrete_obstacles_terrain(terrain, max_height=max_height, min_size=1., max_size=2.,
                                               num_rects=200, platform_size=3.)
                    if choice >= 0.5:
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


def generate_filename(prefix):
    import datetime
    # 获取当前时间
    now = datetime.datetime.now()
    # 将时间格式化为字符串，例如：2022-01-01-12-30-00
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    # 将前缀和时间戳拼接成文件名
    filename = f"{prefix}_{timestamp}.csv"
    return filename