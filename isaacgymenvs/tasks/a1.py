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
import os, time

from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
from .base.vec_task import VecTask

import torch
from typing import Tuple, Dict

from isaacgymenvs.utils.observation_utils import ObservationBuffer

from isaacgymenvs.utils.controller_bridge import SingleControllerBridge, VecControllerBridge


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
        self.xy_velocity_threshold = self.cfg["env"]["xyVelocityCommandThreshold"]

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

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # other
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_flag = self.cfg["env"]["learn"]["pushRobots"]
        self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.add_terrain_obs = self.cfg["env"]["terrain"]["addTerrainObservation"]
        self.num_terrain_obs = self.cfg["env"]["terrain"]["numTerrainObservations"]
        if self.add_terrain_obs:
            self.cfg["env"]["numObservations"] += self.num_terrain_obs

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt  ### wsh_annotation: TODO for what???

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        sensor_forces = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        ### wsh_annotation: use force sensor or not
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)  # shape: num_envs*num_bodies, xyz axis
        self.sensor_forces = gymtorch.wrap_tensor(sensor_forces)

        if self.cfg["env"]["urdfAsset"]["useForceSensor"]:
            self.contact_forces = self.sensor_forces[:, :3].view(self.num_envs, -1,
                                                                 3)  # shape: num_envs, num_bodies, xyz axis
        else:
            self.contact_forces = self.net_contact_forces.view(self.num_envs, -1,
                                                               3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        # self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) ### wsh_annotation
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                    requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                                           device=self.device, requires_grad=False, )
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
                             "action_rate": torch_zeros(), "hip": torch_zeros()}

        self.base_quat = self.root_states[:, 3:7]
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

        ### wsh_annotation: observations dict
        self.obs_name_to_value = {"linearVelocity": self.base_lin_vel,
                                  "angularVelocity": self.base_ang_vel,
                                  "projectedGravity": self.projected_gravity,
                                  "dofPosition": self.dof_pos,
                                  "dofVelocity": self.dof_vel,
                                  "lastAction": self.actions,
                                  "commands": self.commands[:, :3],
                                  "feetContactState": self.feet_contact_state}
        self.obs_combination = self.cfg["env"]["learn"]["observationConfig"]["combination"]
        self.obs_components = self.cfg["env"]["learn"]["observationConfig"]["components"]
        add_obs_noise = self.cfg["env"]["learn"]["observationConfig"]["addNoise"]
        self.obs_buffer_dict = {}
        for key in self.obs_combination.keys():
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

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.compute_observations()
        self.init_done = True

        self.mit_controller = VecControllerBridge(self.num_envs, 16, self.device)

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
            # self.custom_origins = True
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
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
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

            ### wsh_annotation: set friction and restitution of robots
            # for s in range(len(rigid_shape_prop)):
            #     rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]
            #     rigid_shape_prop[s].restitution = restitution_buckets[i % num_buckets]

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

        self.reset_buf[:] = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf),
                                     self.reset_buf)

    def compute_observations(self):  ### TODO(completed) wsh_annotation: add history buffer and delay. contain terrain info or not.
        # calculate omega command
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.commands[:, 2] = self._heading_to_omega(heading)

        self.commands[:, 0] = 0.5
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = 0.0

        ### wsh_annotation: record new observations into buffer
        for key in self.obs_combination.keys():
            self.obs_buffer_dict[key].record(self.obs_name_to_value[key])

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

    def compute_reward(self):
        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        # other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # orientation penalty TODO relating to velocity
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        # base height penalty
        rew_base_height = torch.square(self.root_states[:, 2] - self.desired_base_height) * self.rew_scales["base_height"]  # TODO(completed) add target base height to cfg

        # torque penalty TODO power (torque * motor_speed)
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

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
        rew_air_time = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) * self.rew_scales["air_time"]
        rew_air_time[:] *= torch.norm(self.commands[:, :2], dim=1) > self.xy_velocity_threshold  # no reward for zero command
        self.feet_air_time *= (~(self.feet_contact_state > 0.5)).to(torch.int)

        # cosmetic penalty for hip motion
        rew_hip = torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1) * \
                  self.rew_scales["hip"]

        # total reward
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height + \
                       rew_torque + rew_joint_acc + rew_collision + rew_action_rate + rew_air_time + rew_hip + rew_stumble
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

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

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions_offset = torch_rand_float(0.8, 1.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        # self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        # self.dof_vel[env_ids] = velocities

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.

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
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.xy_velocity_threshold).unsqueeze(1)  # set small commands to zero. wsh_annotation: TODO 0.25 ?

        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        # self.progress_buf[env_ids] = 0
        # self.reset_buf[env_ids] = 1  ### wsh_annotation: TODO

        ### wsh_annotation: TODO(completed) reset to acquire the initial obs_buf
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])

        # calculate the linear acceleration of the base
        self.base_lin_acc[env_ids] = 0.
        self.base_lin_acc[env_ids] -= self.gravity_acc
        self.last_base_lin_vel_rel_world[env_ids] = self.root_states[env_ids, 7:10].clone().detach()

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.feet_contact_state[env_ids] = 1

        self.controller_reset_buf[env_ids] = 1

        ### wsh_annotation: reset observation buffer
        for key in self.obs_combination.keys():
            if key == "commands":  ### wsh_annotation: command history is zero
                self.obs_buffer_dict[key].reset_and_fill_index(env_ids, torch.zeros(len(env_ids), 3, dtype=torch.float, device=self.device, requires_grad=False))
            else:
                self.obs_buffer_dict[key].reset_and_fill_index(env_ids, self.obs_name_to_value[key][env_ids])

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

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
        self.root_states[:, 7:9] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def pre_physics_step(self, actions):
        ### wsh_annotation: TODO feed forward torque
        # self.actions[:] = actions.clone().to(self.device)
        self.actions[:, [0, 3, 6, 9]] = 0.
        dof_pos_desired = self.action_scale * self.actions + self.default_dof_pos
        torques = torch.zeros_like(self.torques)
        for i in range(self.decimation - 1):
            torques[:] = self.mit_controller.get_torque(self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state) # self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state
            # torques = torch.clip(self.Kp * (dof_pos_desired - self.dof_pos) - self.Kd * self.dof_vel, -33.5, 33.5)
            # print("torques: ", torques)
            # torques = torch.zeros_like(torques)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            # self.torques = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            # self.gym.refresh_dof_state_tensor(self.sim)
            self.update_pre_state()
        torques[:] = self.mit_controller.get_torque(self.controller_reset_buf, self.base_quat, self.base_ang_vel, self.base_lin_acc, self.dof_pos, self.dof_vel, self.feet_contact_state)
        # torques = torch.clip(self.Kp * (dof_pos_desired - self.dof_pos) - self.Kd * self.dof_vel, -33.5, 33.5)
        # print("torques: ", torques)
        # torques = torch.zeros_like(torques)
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
        self.torques = torques.view(self.torques.shape)

    def post_physics_step(self):
        # self.gym.refresh_dof_state_tensor(self.sim)
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)

        self.progress_buf += 1
        self.randomize_buf += 1
        self.common_step_counter += 1
        if self.push_flag and self.common_step_counter % self.push_interval == 0:  ### wsh_annotation: self.push_interval > 0
            self.push_robots()

        # # prepare quantities
        # self.base_quat[:] = self.root_states[:, 3:7]
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.update_pre_state()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()

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

    def update_pre_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.feet_contact_state[:] = self.contact_forces[:, self.feet_indices, 2] > self.stance_foot_force_threshold

        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, ((self.root_states[:,
                                                                     7:10] - self.last_base_lin_vel_rel_world) / self.sim_params.dt - self.gravity_acc))
        self.last_base_lin_vel_rel_world[:] = self.root_states[:, 7:10].clone().detach()

        self.controller_reset_buf[:] = 0

        # print("lin acc: ", self.base_lin_acc)
        # print("last vel:", self.last_base_lin_vel_rel_world)

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
            if not self.add_terrain_obs:
                if choice < 0.0:
                    pass
                elif choice < 0.4:
                    random_uniform_terrain(terrain, min_height=-0.03 * uniform_difficulty,
                                           max_height=0.03 * uniform_difficulty, step=0.05, downsampled_scale=0.5)
                elif choice < 0.7:
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
