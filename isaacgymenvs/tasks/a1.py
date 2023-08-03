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


class A1(VecTask):

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
        self.desired_base_height = pos[2]

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
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        self.rew_scales["stumble"] = self.cfg["env"]["learn"]["feetStumbleRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["energy"] = self.cfg["env"]["learn"]["energyRewardScale"]
        self.rew_scales["power"] = self.cfg["env"]["learn"]["powerRewardScale"]
        self.rew_scales["power_max_mean_each"] = self.cfg["env"]["learn"]["power_max_mean_each"]
        self.rew_scales["power_max_mean_std"] = self.cfg["env"]["learn"]["power_max_mean_std"]
        self.rew_scales["feet_max_force_total"] = self.cfg["env"]["learn"]["feet_max_force_total"]
        self.rew_scales["feet_max_force_std"] = self.cfg["env"]["learn"]["feet_max_force_std"]
        self.rew_scales["torque_max_mean_each"] = self.cfg["env"]["learn"]["torque_max_mean_each"]
        self.rew_scales["torque_max_mean_std"] = self.cfg["env"]["learn"]["torque_max_mean_std"]

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

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
        if self.add_terrain_obs:
            self.cfg["env"]["numObservations"] += self.num_terrain_obs

        # for key in self.rew_scales.keys():
        #     self.rew_scales[key] *= self.dt  ### wsh_annotation: TODO for what???

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_flag = self.cfg["env"]["learn"]["pushRobots"]
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt  ### wsh_annotation: TODO for what???

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
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
        # self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) ### wsh_annotation
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                    requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                                           device=self.device, requires_grad=False, )
        self.xy_velocity_threshold = torch.tensor(xy_velocity_threshold_list, dtype=torch.float, device=self.device, requires_grad=False)

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                   requires_grad=False)

        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                   requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)

        self.feet_contact_state = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) ### wsh_annotation: 1->contact

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
                             "orient": torch_zeros(), "torques": torch_zeros(), "joint_acc": torch_zeros(),
                             "base_height": torch_zeros(),
                             "air_time": torch_zeros(), "collision": torch_zeros(), "stumble": torch_zeros(),
                             "action_rate": torch_zeros(), "energy": torch_zeros(), "power": torch_zeros(), "hip": torch_zeros(),
                             "power_max_mean_each": torch_zeros(), "power_max_mean_std": torch_zeros(),
                             "feet_max_force_total": torch_zeros(), "feet_max_force_std": torch_zeros(),
                             "torque_max_mean_each": torch_zeros(), "torque_max_mean_std": torch_zeros()}

        self.base_quat = self.root_states[:, 3:7]
        self.euler_xyz = get_euler_xyz2(self.base_quat)
        self.euler_roll = self.euler_xyz.view(self.num_envs, 1, 3)[..., 0]
        self.euler_pitch = self.euler_xyz.view(self.num_envs, 1, 3)[..., 1]
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
        self.motion_planning_cmd = torch.zeros(self.num_envs, 28, dtype=torch.float, device=self.device, requires_grad=False)

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
                                  "lastAction": self.actions,
                                  "commands": self.commands[:, :3],
                                  "feetContactState": self.feet_contact_state,
                                  "bodyPos": self.root_states[:, :3],
                                  "motorTorque": self.torques,
                                  "feetForce": self.feet_force,
                                  "dofPositionRelInit": self.dof_pos_rel_init,
                                  "rollAngle": self.euler_roll,
                                  "pitchAngle": self.euler_pitch}
        self.obs_combination = self.cfg["env"]["learn"]["observationConfig"]["combination"]
        self.obs_components = self.cfg["env"]["learn"]["observationConfig"]["components"]
        add_obs_noise = self.cfg["env"]["learn"]["observationConfig"]["addNoise"]
        self.obs_buffer_dict = {}
        self.record_items = self.obs_components.keys()
        for key in self.record_items:
            if add_obs_noise:
                noise = self.obs_components[key]["noise"]
            else:
                noise = None
            self.obs_buffer_dict[key] = ObservationBuffer(num_envs=self.num_envs,
                                                          single_data_shape=(self.obs_components[key]["size"],),
                                                          data_type=torch.float,
                                                          buffer_length=self.obs_components[key]["bufferLength"],
                                                          device=self.device,
                                                          scale=self.obs_components[key]["scale"],
                                                          noise=noise)

        self.vel_average = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)

        self.motion_planning_interface = MotionPlanningInterface(self.num_envs, 28, self.device)
        self.gait_period_offset = torch.zeros_like(self.feet_contact_state)
        self.gait_duty_cycle_offset = torch.zeros_like(self.feet_contact_state)
        self.gait_phase_offset = torch.zeros_like(self.feet_contact_state)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # self.compute_observations()
        self.init_done = True

        self.mit_controller = VecControllerBridge(self.num_envs, self.cfg["num_controller_threads"], self.device)
        # self.motion_planning_interface = MotionPlanningInterface(self.num_envs, 28, self.device)

        self.record_data = np.expand_dims(np.arange(57), axis=0)
        self.record_data_test = np.expand_dims(np.arange(116), axis=0)
        self.record_path = ''

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
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"][
            "collapseFixedJoints"]  ### wsh_annotation: modofy 'True' to cfg parameter
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False
        asset_options.vhacd_enabled = True

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
        if self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]:
            foot_name = calf_name
        thigh_names = [s for s in body_names if thigh_name in s]
        feet_names = [s for s in body_names if foot_name in s]

        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        # acquire link indices
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_asset_rigid_body_index(a1_asset, thigh_names[i])
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_asset_rigid_body_index(a1_asset, feet_names[i])
        self.base_index = self.gym.find_asset_rigid_body_index(a1_asset, base_name)

        # get assert props
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(a1_asset)
        dof_props = self.gym.get_asset_dof_properties(a1_asset)
        for i in range(self.num_dof):
            dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT  # gymapi.DOF_MODE_POS
            # dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"] #self.Kp
            # dof_props['damping'][i] = self.cfg["env"]["control"]["damping"] #self.Kd
            # dof_props['armature'][i] = 0.05

        # prepare friction & restitution randomization
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        restitution_range = self.cfg["env"]["learn"]["restitutionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device=self.device)
        restitution_buckets = torch_rand_float(restitution_range[0], restitution_range[1], (num_buckets, 1),
                                               device=self.device)

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
            spacing = 0.  ### wsh_annotation: the same origin (0., 0.) of env coordinates

        # prepare for env creatation
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.a1_handles = []
        self.envs = []

        # env creating
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            a1_handle = self.gym.create_actor(env_handle, a1_asset, start_pose, "a1", i, 0, 0)

            ## wsh_annotation: set friction and restitution of robots
            for s in range(len(rigid_shape_prop)):
                # rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
                # rigid_shape_prop[s].restitution = restitution_buckets[i % num_buckets]
                rigid_shape_prop[s].friction = 1.
                rigid_shape_prop[s].rolling_friction = 0.
                rigid_shape_prop[s].torsion_friction = 0.
                rigid_shape_prop[s].restitution = self.cfg["env"]["terrain"]["restitution"]
                rigid_shape_prop[s].contact_offset = 0.001
                rigid_shape_prop[s].rest_offset = -0.003

            self.gym.set_actor_rigid_shape_properties(env_handle, a1_handle, rigid_shape_prop)

            self.gym.set_actor_dof_properties(env_handle, a1_handle, dof_props)

            rigid_body_prop = self.gym.get_actor_rigid_body_properties(env_handle, a1_handle)

            self.envs.append(env_handle)
            self.a1_handles.append(a1_handle)

    def _heading_to_omega(self, heading):
        return torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.5, 1.5)

    def check_termination(self):
        self.reset_buf[:] = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.
        if not self.allow_knee_contacts:
            knee_contact = torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > 1.
            self.reset_buf[:] |= torch.any(knee_contact, dim=1)

        # pos limit termination
        # self.reset_buf[:] |= self.root_states[:, 2] < 0.28
        # self.reset_buf[:] |= torch.abs(self.euler_xyz[:, 0]) > 0.2
        # self.reset_buf[:] |= torch.abs(self.euler_xyz[:, 1]) > 0.4

        self.timeout_buf[:] = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf == 0)
        self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf),
                                     self.reset_buf)
        # print(knee_contact)

    def compute_observations(self):  ### TODO(completed) wsh_annotation: add history buffer and delay. contain terrain info or not.


        # ### wsh_annotation: record new observations into buffer
        # for key in self.record_items:
        #     self.obs_buffer_dict[key].record(self.obs_name_to_value[key])

        self.commands[:, 0] = 0.
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = 0.0
        # self.modify_vel_command()
        # self.commands[:, 0] = 2.5

        self.record_commands()

        tmp_obs_buf = torch.cat([self.obs_buffer_dict[key].get_index_data(self.obs_combination[key]) for key in self.obs_combination.keys()], dim=-1)

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

        if self.add_terrain_obs:
            self.measured_heights = self.get_heights()
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.) * self.height_meas_scale
            self.obs_buf[:] = torch.cat((tmp_obs_buf, heights), dim=-1)
        else:
            self.obs_buf[:] = tmp_obs_buf[:]

        # self.obs_buf[:] = torch.cat((self.base_lin_vel*self.lin_vel_scale,
        #                              self.base_ang_vel*self.ang_vel_scale,
        #                              self.projected_gravity,
        #                              self.commands[:, :3]*torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], requires_grad=False, device=self.device),
        #                              self.dof_pos_rel_init*self.dof_pos_scale,
        #                              self.dof_vel*self.dof_vel_scale,
        #                              self.actions
        #                              ), dim=-1)

    def compute_reward(self):
        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        # lin_vel_error = torch.sum(self.base_lin_vel_error_square_accumulated[:, :2], dim=1) / self.decimation
        # ang_vel_error = self.base_ang_vel_error_square_accumulated[:, 2] / self.decimation
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

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
        # rew_base_height = torch.square(self.root_states[:, 2] - self.desired_base_height) * self.rew_scales["base_height"]  # TODO(completed) add target base height to cfg
        base_height_error = torch.square((self.root_states[:, 2] - self.desired_base_height) * 1000)
        rew_base_height = (1. - torch.exp(-base_height_error / 10.)) * self.rew_scales["base_height"]
        # print(self.root_states[0, 2])

        # torque penalty TODO power (torque * motor_speed)
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]
        # rew_torque = torch.sum(self.torques_square_accumulated, dim=1) / self.decimation * self.rew_scales["torque"]
        # self.torques_square_accumulated[:] = 0.0

        # joint acc penalty
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - self.dof_vel), dim=1) * self.rew_scales["joint_acc"]

        # collision penalty
        knee_contact = torch.norm(self.contact_forces[:, self.thigh_indices, :], dim=2) > self.contact_force_threshold
        rew_collision = torch.sum(knee_contact, dim=1) * self.rew_scales["collision"]  # sum vs any ?

        # stumbling penalty TODO contact forces x & y are inaccurate
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5.) * (
                    torch.abs(self.contact_forces[:, self.feet_indices, 2]) < self.contact_force_threshold)
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"]

        # action rate penalty TODO action is uncertain
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]

        # air time reward
        # contact = torch.norm(contact_forces[:, feet_indices, :], dim=2) > 1.
        # self.feet_contact_state[:] = self.contact_forces[:, self.feet_indices, 2] > self.stance_foot_force_threshold
        first_contact = (self.feet_air_time > 0.) * self.feet_contact_state
        self.feet_air_time += self.dt
        # reward only on first contact with the ground TODO self.feet_air_time - 0.5 ?
        rew_air_time = torch.sum((self.feet_air_time - 0.15) * first_contact, dim=1) * self.rew_scales["air_time"]
        # rew_air_time[:] *= torch.norm(self.commands[:, :2], dim=1) > self.xy_velocity_threshold  # no reward for zero command
        rew_air_time[:] *= ~((self.commands[:, :3].abs() < self.xy_velocity_threshold).all(dim=-1))
        self.feet_air_time *= (~(self.feet_contact_state > 0.5)).to(torch.int)

        # energy efficiency penalty
        energy = self.torques * self.dof_vel
        # energy += 0.3 * torch.square(self.torques)
        energy = torch.clip(energy, min=0., max=None)
        energy = torch.sum(energy, dim=1)
        tmp_lin_v = torch.norm(self.base_lin_vel, dim=1)
        energy_cot = torch.where(tmp_lin_v > 0, energy / (12.776 * 9.8 * tmp_lin_v), energy / 4.0)
        rew_energy = (1. - torch.exp(-energy_cot / 0.25)) * self.rew_scales["energy"]

        # cosmetic penalty for hip motion
        rew_hip = torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1) * \
                  self.rew_scales["hip"]

        # survival reward, ensures that the reward is always positive
        # rew_survival = self.progress_buf / (self.max_episode_length - 1)
        # rew_survival = 20.0 * self.commands[0, 0]
        rew_survival = 0.15

        buf_length = 4
        body_height_buf = self.obs_buffer_dict["bodyPos"].get_len_data_raw(buf_length)[:, 2, :]
        lin_vel_buf = self.obs_buffer_dict["linearVelocity"].get_len_data_raw(buf_length)
        ang_vel_buf = self.obs_buffer_dict["angularVelocity"].get_len_data_raw(buf_length)
        projectedGravity_buf = self.obs_buffer_dict["projectedGravity"].get_len_data_raw(buf_length)
        motor_vel_buf = self.obs_buffer_dict["dofVelocity"].get_len_data_raw(buf_length)
        motor_torque_buf = self.obs_buffer_dict["motorTorque"].get_len_data_raw(buf_length)
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
        torque_mean_each = torch.norm(motor_torque_buf, dim=-1)
        torque_max_mean_each = torch.mean(torque_mean_each, dim=-1)
        torque_max_mean_std = torch.std(torque_mean_each[:, [1, 2, 4, 5, 7, 8, 10, 11]], dim=-1)

        tmp_vel_mean = torch.stack([v_x_mean, v_y_mean, v_z_mean, v_roll_mean, v_pitch_mean, v_yaw_mean], dim=1)
        self.vel_average[:] = tmp_vel_mean * 0.2 + self.vel_average * 0.8
        # print(f"vx_average: {self.vel_average[:, 0]}")


        # print(f"v_x_mean: {v_x_mean}")
        # print(f"v_y_mean: {v_y_mean}")
        # print(f"v_yaw_mean: {v_yaw_mean}")
        # print(self.obs_buffer_dict["linearVelocity"].get_latest_data_raw()[0][0])

        rew_base_height = torch.square(body_height_mean - self.desired_base_height) * self.rew_scales["base_height"]

        lin_vel_error = torch.square(self.commands[:, 0] - self.vel_average[:, 0]) + torch.square(self.commands[:, 1] - self.vel_average[:, 1])
        # rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_lin_vel_xy = -lin_vel_error * self.rew_scales["lin_vel_xy"]
        ang_vel_error = torch.square(self.commands[:, 2] - self.vel_average[:, 5])
        # rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]
        rew_ang_vel_z = -ang_vel_error * self.rew_scales["ang_vel_z"]
        rew_power = power_mean_total * self.rew_scales["power"]

        rew_power_max = power_max_mean_each * self.rew_scales["power_max_mean_each"]
        rew_power_max_std = power_max_mean_std * self.rew_scales["power_max_mean_std"]
        rew_feet_force_max = feet_max_force_total * self.rew_scales["feet_max_force_total"]
        rew_feet_force_max_std = feet_max_force_std * self.rew_scales["feet_max_force_std"]
        rew_torque_max = torque_max_mean_each * self.rew_scales["torque_max_mean_each"]
        rew_torque_max_std = torque_max_mean_std * self.rew_scales["torque_max_mean_std"]

        # rew_lin_vel_xy = -20 * torch.abs(self.vel_average[:, 0] - self.commands[:, 0]) - torch.square(self.vel_average[:, 1] - self.commands[:, 1])
        # rew_ang_vel_z = -1.0 * torch.square(self.vel_average[:, 5] - self.commands[:, 2])
        # rew_energy = -0.04 * energy


        # total reward
        self.rew_buf[:] = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height + \
                       rew_torque + rew_joint_acc + rew_collision + rew_action_rate + rew_air_time + rew_hip + rew_stumble + rew_energy + rew_power + rew_survival + \
                       rew_power_max + rew_power_max_std + rew_feet_force_max + rew_feet_force_max_std + rew_torque_max + rew_torque_max_std

        # self.rew_buf[:] = rew_lin_vel_xy + rew_ang_vel_z + rew_energy + rew_survival
        # print(self.rew_buf)
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
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["collision"] += rew_collision
        self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["air_time"] += rew_air_time
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["hip"] += rew_hip
        self.episode_sums["energy"] += rew_energy
        self.episode_sums["power"] += rew_power
        self.episode_sums["power_max_mean_each"] += rew_power_max
        self.episode_sums["power_max_mean_std"] += rew_power_max_std
        self.episode_sums["feet_max_force_total"] += rew_feet_force_max
        self.episode_sums["feet_max_force_std"] += rew_feet_force_max_std
        self.episode_sums["torque_max_mean_each"] += rew_torque_max
        self.episode_sums["torque_max_mean_std"] += rew_torque_max_std

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
        self.commands[env_ids, 3] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1],
                                                     (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 2] = self.commands[env_ids, 3]
        # self.commands[env_ids, 1] = 0.
        # self.commands[env_ids, 2] = 0.
        # self.commands[env_ids, 0] = 1.0
        # self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.xy_velocity_threshold).unsqueeze(1)  # set small commands to zero. wsh_annotation: TODO 0.25 ?
        # self.modify_vel_command()
        # self.commands[env_ids] *= ~((self.commands[env_ids, :3].abs() < self.xy_velocity_threshold).all(dim=-1)).unsqueeze(1)
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

        # calculate the linear acceleration of the base
        self.base_lin_acc[env_ids] = 0.
        self.base_lin_acc[env_ids] -= self.gravity_acc
        self.last_base_lin_vel_rel_world[env_ids] = self.root_states[env_ids, 7:10].clone().detach()

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.feet_contact_state[env_ids] = 1
        self.feet_force[env_ids] = self.contact_forces[env_ids][:, self.feet_indices].reshape(len(env_ids), -1)

        self.controller_reset_buf[env_ids] = 1

        # self.gait_period_offset[env_ids] = (torch.zeros_like(self.feet_contact_state) + torch_rand_float(-0.23, 0.1, (self.num_envs, 1), device=self.device))[env_ids]
        self.gait_period_offset[env_ids] = -0.2
        self.gait_duty_cycle_offset[env_ids] = (torch.zeros_like(self.feet_contact_state) - 0.0)[env_ids]
        self.gait_phase_offset[env_ids] = (torch.zeros_like(self.feet_contact_state))[env_ids]

        self.motion_planning_interface.update_gait_planning(True, self.gait_period_offset, self.gait_duty_cycle_offset,
                                                            self.gait_phase_offset, None)
        self.motion_planning_interface.update_body_planning(True, None, None, None, None, self.commands[:, :3])
        self.motion_planning_interface.generate_motion_command()

        # self.commands[env_ids, 0] = 1.0
        # self.commands[env_ids, 1] = 0.0
        # self.commands[env_ids, 2] = 0.0

        ### wsh_annotation: reset observation buffer
        for key in self.record_items:
            if key == "commands":  ### wsh_annotation: command history is zero
                self.obs_buffer_dict[key].reset_and_fill_index(env_ids, torch.zeros(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False))
            else:
                self.obs_buffer_dict[key].reset_and_fill_index(env_ids, self.obs_name_to_value[key][env_ids])

        self.vel_average[env_ids] = 0.0

        self.compute_observations()

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids] / (self.progress_buf[env_ids] * self.dt + 1.e-5))  # / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
            # print(self.extras["episode"]['rew_' + key])
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

        self.progress_buf[env_ids] = 0

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        ### wsh_annotation: TODO 卡BUG loop in last and current level ??
        self.terrain_levels[env_ids] -= 1 * (distance < torch.norm(
            self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25)  ### wsh_annotation: TODO 0.25 ?
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def push_robots(self):
        # wsh_annotation: TODO(!!!) How about add external forces ??
        # self.root_states[:, 7:9] = torch_rand_float(-1.2, 1.2, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = torch.tensor([2.5, 4.0], device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def pre_physics_step(self, actions):
        # print(self.commands)
        ### wsh_annotation: TODO feed forward torque
        self.actions[:] = actions.clone().to(self.device)
        # dof_pos_desired = self.action_scale * self.actions + self.default_dof_pos
        # print(self.actions[:10])
        # self.actions[:, [0, 3, 6, 9]] = 0.
        dof_pos_desired = self.actions.clone()
        dof_pos_desired[:, [0, 3, 6, 9]] *= 0.15
        dof_pos_desired[:, [1, 4, 7, 10]] *= 0.4
        dof_pos_desired[:, [2, 5, 8, 11]] *= 0.4
        # dof_pos_desired[:, [0, 3, 6, 9]] *= 0.8
        # dof_pos_desired[:, [1, 4, 7, 10]] = 1.325 * dof_pos_desired[:, [1, 4, 7, 10]] - 0.525
        # dof_pos_desired[:, [2, 5, 8, 11]] = 0.887 * dof_pos_desired[:, [2, 5, 8, 11]] - 0.213
        # dof_pos_desired = self.action_scale * self.actions
        dof_pos_desired += self.default_dof_pos
        # dof_pos_desired = self.default_dof_pos
        # tt = 15. * self.actions

        gait_period_offset = torch.zeros_like(self.feet_contact_state)-0.1
        # gait_period_offset = self.actions[:, :4] * 0.3
        # gait_duty_cycle_offset = self.actions[:, :4] * 0.5
        # gait_phase_offset = self.actions[:, 4:8] * 0.5
        # gait_phase_offset[:, 1:3] += 0.5
        # body_height_offset = self.actions * 0.01
        # body_orientation = torch.zeros_like(self.base_ang_vel)
        # body_orientation[:, :2] = self.actions[:, 1:] * 0.2
        # body_height_offset[:] = 0.0
        # print(f"body_height_offset: {body_height_offset*1000} mm")
        # print(f"body_height_real: {(self.root_states[:, 2] - 0.3) * 1000} mm")

        # gait_period_offset = torch.zeros_like(self.feet_contact_state) + (torch.rand(self.num_envs, 1, dtype=float, device=self.device) * 2 - 1) * 0.2
        gait_duty_cycle_offset = torch.zeros_like(self.feet_contact_state)-0.0
        gait_phase_offset = torch.zeros_like(self.feet_contact_state)

        # body_height_offset = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # body_orientation = torch.zeros_like(self.base_ang_vel)

        self.motion_planning_interface.update_gait_planning(True, gait_period_offset, gait_duty_cycle_offset, gait_phase_offset, None)
        self.motion_planning_interface.update_body_planning(True, None, None, None, None, self.commands[:, :3])
        self.motion_planning_interface.generate_motion_command()

        # print(self.base_lin_vel[0][0])
        # print("gait_duty_cycle", (gait_duty_cycle_offset + 0.5)[0])
        # print("gait_phase_offset", gait_phase_offset[0])

        # torques = torch.zeros_like(self.torques)
        for i in range(self.decimation - 1):
            torques = self.mit_controller.get_torque(self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state, self.motion_planning_interface.get_motion_command()) # self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state
            self.motion_planning_interface.change_gait_planning(False)
            self.motion_planning_interface.change_body_planning(False)
            # torques[:] = tt

            # torques = torch.clip(self.Kp * (dof_pos_desired - self.dof_pos) - self.Kd * self.dof_vel, -33.5, 33.5)
            # torques = torch.clip(torques, -33.5, 33.5)

            tmp_max_torque = torch.clip(79.17 - 3.953886 * self.dof_vel, 0, 33.5)
            tmp_min_torque = torch.clip(-79.17 - 3.953886 * self.dof_vel, -33.5, 0)
            torques[:] = torch.where(self.dof_vel > 11.55, torch.clip(torques, -33.5 * torch.ones_like(torques), tmp_max_torque), torques)
            torques[:] = torch.where(self.dof_vel < -11.55, torch.clip(torques, tmp_min_torque, 33.5 * torch.ones_like(torques)), torques)
            # print("torques: ", torques)
            # torques = torch.zeros_like(torques)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques[:] = torques.view(self.torques.shape)
            # self.torques = torques.view(self.torques.shape)
            # if i % 10 == 0 and self.force_render:
            #     self.render()
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            # self.gym.refresh_dof_state_tensor(self.sim)
            # self.record_state_test()

            self.update_pre_state()
            self.record_states_into_buffer()
            self.record_commands()

            # self.torques_square_accumulated += torch.square(torques)

        torques[:] = self.mit_controller.get_torque(self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state, self.motion_planning_interface.get_motion_command())
        # torques[:] = tt

        # torques = torch.clip(self.Kp * (dof_pos_desired - self.dof_pos) - self.Kd * self.dof_vel, -33.5, 33.5)
        # torques = torch.clip(torques, -33.5, 33.5)
        tmp_max_torque = torch.clip(79.17 - 3.953886 * self.dof_vel, 0, 33.5)
        tmp_min_torque = torch.clip(-79.17 - 3.953886 * self.dof_vel, -33.5, 0)
        torques[:] = torch.where(self.dof_vel > 11.55,
                                 torch.clip(torques, -33.5 * torch.ones_like(torques), tmp_max_torque), torques)
        torques[:] = torch.where(self.dof_vel < -11.55,
                                 torch.clip(torques, tmp_min_torque, 33.5 * torch.ones_like(torques)), torques)
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
        # if self.push_flag and self.common_step_counter % self.push_interval == 0:  ### wsh_annotation: self.push_interval > 0
        #     self.push_robots()

        # # prepare quantities
        # self.base_quat[:] = self.root_states[:, 3:7]
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.update_pre_state()
        self.record_states_into_buffer()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids)

        # modify vel command
        # self.modify_vel_command()
        self.compute_observations()

        if self.push_flag and self.common_step_counter % self.push_interval == 0:  ### wsh_annotation: self.push_interval > 0
            self.push_robots()

        ### wsh_annotation
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            # draw height lines
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))

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

    # ### wsh_annotation: get observation value
    # def get_base_lin_vel(self):
    #     return self.base_lin_vel
    #
    # def get_base_ang_vel(self):
    #     return self.base_ang_vel

    def record_state(self):
        record_path = "/home/wsh/Documents/record_data/imitation/imitation_mpc_data_walk_2.csv"
        if self.common_step_counter == 0:
            if os.path.exists(record_path):
                os.remove(record_path)
            with open(record_path, 'a+') as fp:
                fp.write("gait type: walk, duty: 0.5, phase offset: (0, 0.3, 0.5, 0.8)" + '\n')
                fp.write("(0 ~ 3): gait period(s)" + '\n')
                fp.write("(0 ~ 3): gait period(s)" + '\n')
                fp.write("(4 ~ 6): body linear velocity(m/s)" + '\n')
                fp.write("(7 ~ 9): body angular velocity(rad/s)" + '\n')
                fp.write("(10~13): body quaternion(x, y, z, w)" + '\n')
                fp.write("(14~25): joint position(rad)" + '\n')
                fp.write("(26~37): joint velocities(rad/s)" + '\n')
                fp.write("(38~41): feet contact state(1->contact)" + '\n')
                fp.write("(42~44): velocity command(vx, vy, omega)" + '\n')
                fp.write("(45~56): torque command(Nm)" + '\n\n')

        data = torch.cat((self.gait_period_offset + 0.5,
                          self.base_lin_vel,
                          self.base_ang_vel,
                          self.base_quat,
                          self.dof_pos,
                          self.dof_vel,
                          self.feet_contact_state,
                          self.commands[:, :3],
                          self.torques), dim=-1).cpu().numpy()
        self.record_data = np.concatenate((self.record_data, data), axis=0)

        if self.common_step_counter >= 500 - 1:
            with open(record_path, 'a+') as fp:
                np.savetxt(fp, self.record_data, delimiter=",")
            exit(0)

    def record_state_test(self):
        if self.common_step_counter == 0 and self.record_path == '':
            record_dir = os.path.join('/home/wsh/Documents/pyProjects/IsaacGymEnvs/isaacgymenvs/runs', 'A1Test_2023-07-13_21-40-24/record_test_data')
            os.makedirs(record_dir, exist_ok=True)
            # file_name = generate_filename('record_data_v-0_3_0')
            file_name = 'record_data-v_0_3_0-A1Test_2023-07-13_21-40-24.csv'
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
                fp.write("[104:116]: torque command(Nm)" + '\n\n')

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
                          self.torques), dim=-1).cpu().numpy()
        self.record_data_test = np.concatenate((self.record_data_test, data), axis=0)

        if self.common_step_counter >= 1500 - 1:
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
        self.projected_gravity[:] = quat_rotate(self.base_quat, self.gravity_vec)

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

        self.feet_force[:] = self.contact_forces[:, self.feet_indices].view(self.num_envs, -1)
        self.feet_contact_state[:] = self.contact_forces[:, self.feet_indices, 2] > self.stance_foot_force_threshold

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
        # calculate omega command
        # forward = quat_apply(self.base_quat, self.forward_vec)
        # heading = torch.atan2(forward[:, 1], forward[:, 0])
        # self.commands[:, 2] = self._heading_to_omega(heading)

        # self.commands[:, 0] = 1.0
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = 0.0

        n = 1500  # 3000 # 100
        count = self.common_step_counter % n
        if count > n // 2:
            count = n - count
        self.commands[:, 0] = 0.004 * count  # 0.002 # 0.06
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

    def record_states_into_buffer(self):
        ### wsh_annotation: record new states into buffer
        for key in self.record_items:
            if key != "commands":
                self.obs_buffer_dict[key].record(self.obs_name_to_value[key])

    def record_commands(self):
        self.obs_buffer_dict["commands"].record(self.obs_name_to_value["commands"])


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
            uniform_difficulty = 1
            if not self.add_terrain_obs:
                if choice < 0.0:
                    pass
                elif choice < -1.4:
                    random_uniform_terrain(terrain, min_height=-0.03 * uniform_difficulty,
                                           max_height=0.03 * uniform_difficulty, step=0.05, downsampled_scale=0.2)
                elif choice < 1.7:
                    slope = 0.2 * difficulty
                    if choice < 0.5:
                        slope *= -1
                    pyramid_sloped_terrain(terrain, slope=slope, platform_size=0.5)
                    if np.random.choice([0, 1]):
                        random_uniform_terrain(terrain, min_height=-0.005 * uniform_difficulty,
                                               max_height=0.005 * uniform_difficulty, step=0.05, downsampled_scale=0.5)
                # elif choice < 0.8:
                #     step_height = 0.05 * difficulty
                #     if choice < 0.7:
                #         step_height *= -1
                #     pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
                #     if np.random.choice([0, 1]):
                #         random_uniform_terrain(terrain, min_height=-0.002 * uniform_difficulty,
                #                                max_height=0.002 * uniform_difficulty, step=0.05, downsampled_scale=0.5)
                else:
                    max_height = 0.03 * difficulty
                    discrete_obstacles_terrain(terrain, max_height=max_height, min_size=1., max_size=2., num_rects=200,
                                               platform_size=3.)
                    if np.random.choice([0, 1]):
                        random_uniform_terrain(terrain, min_height=-0.005 * uniform_difficulty,
                                               max_height=0.005 * uniform_difficulty, step=0.05, downsampled_scale=0.5)

            else:
                if choice < 0.1:
                    if np.random.choice([0, 1]):
                        pyramid_sloped_terrain(terrain, np.random.choice(
                            [-0.3, -0.2, 0, 0.2, 0.3]))  ### wsh_annotation: slope_angle = arc-tan(slope)
                        random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05,
                                               downsampled_scale=0.2)
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
            difficulty = (i + 1) / (num_levels) #  - 1
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
