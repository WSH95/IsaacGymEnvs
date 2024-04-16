# -*- coding: utf-8 -*-
# Created by Shuhan Wang on 2023/11/10.
#

import os
import sys
from isaacgym import gymapi, gymutil, gymtorch
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, normalize, quat_rotate, quat_apply, quat_rotate_inverse, copysign
import torch
import numpy as np
from tqdm import tqdm
from isaacgymenvs.utils.controller_bridge import VecControllerBridge
from isaacgymenvs.utils.motion_planning_interface import MotionPlanningInterface
import time
from datetime import datetime
import progressbar

NUM_ENVS = 16
HEADLESS = False
USE_GPU = True
SIM_DT = 0.002
DECIMATION = 10
MAX_EPISODE_LEN_S = 200.0
SUB_STEPS = 1
ASSERT_PATH = "/home/wsh/Documents/pyProjects/IsaacGymEnvs/assets/urdf/a1/urdf/a1_old.urdf"
BASE_NAME = 'trunk'
THIGH_NAME = 'thigh'
CALF_NAME = 'calf'
FOOT_NAME = 'foot'
BODY_HEIGHT_INIT = 0.3
DEFAULT_DOF_POS = [0.01, 0.7954, -1.5908, -0.01, 0.7954, -1.5908, 0.01, 0.7954, -1.5908, -0.01, 0.7954, -1.5908]
SPACING = 3.0
VIEWER_POS_INIT = [-3, -3., 3.]
VIEWER_LOOKAT_INIT = [1., 1., 0.]
PUSH_FLAG = False
PUSH_INTERVAL_S = 3.0


class A1Env:
    def __init__(self):
        self.num_envs = NUM_ENVS
        self.headless = HEADLESS
        self.decimation = DECIMATION
        self.sim_dt = SIM_DT
        self.dt = self.sim_dt * self.decimation
        self.max_episode_length = int(MAX_EPISODE_LEN_S / self.dt + 0.5)

        self.push_flag = PUSH_FLAG
        self.push_interval = int(PUSH_INTERVAL_S / self.dt + 0.5)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.render_fps: int = -1
        self.last_frame_time: float = 0.0
        self.control_freq_inv = 1

        self.gym = gymapi.acquire_gym()
        self.physics_engine = gymapi.SIM_PHYSX
        self._create_sim()
        self._create_ground_plane()
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._set_viewer()
        self._init_buffers()
        self.motion_planning_interface = MotionPlanningInterface(self.num_envs, 56, self.device)
        self.mit_controller = VecControllerBridge(self.num_envs, 26, self.device)
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def run(self):
        while True:
            self.step()

    def step(self):
        if self.common_step_counter % 1 == 0:
            self.render()

        force_ff_mpc, torques_est, tau_ff_mpc, q_des, qd_des = self._cal_torque()

        for i in range(self.decimation):
            # torques, _, _, _ = self._cal_torque()
            torques = tau_ff_mpc
            # torques = self._cal_pd(tau_ff_mpc, q_des, qd_des, kp=20.0, kd=0.5)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques[:] = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self._update_pre_state()

            if i < self.decimation - 1:
                _, _, tau_ff_mpc, _, _ = self._cal_torque()

        self.post_physics_step()

    def post_physics_step(self):
        self.progress_buf[:] += 1
        self.common_step_counter += 1

        if self.push_flag and self.common_step_counter % self.push_interval == 0:  ### wsh_annotation: self.push_interval > 0
            self.push_robots()

        # print('base linear velocity_x: ', self.base_lin_vel[0, 0])
        # print('base linear velocity_y: ', self.base_lin_vel[0, 1])
        print('base linear velocity_w: ', self.base_ang_vel[0, 2])

        self.check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

    def check_termination(self):
        self.timeout_buf[:] = torch.where(self.progress_buf > self.max_episode_length - 1,
                                          torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))
        collision = torch.norm(self.contact_forces[:, self.penalize_contacts_indices, :], dim=2) > 2.0
        collision2 = torch.any(collision, dim=1)
        roll_over = torch.abs(self.euler_xyz[:, 0]) > 0.5
        pitch_over = torch.abs(self.euler_xyz[:, 1]) > 0.7

        self.reset_buf[:] = collision2.clone()
        self.reset_buf[:] |= roll_over
        self.reset_buf[:] |= pitch_over
        self.timeout_buf[:] = torch.where(self.reset_buf.bool().clone(), torch.zeros_like(self.timeout_buf), self.timeout_buf)
        self.reset_buf[:] = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

    def _cal_torque(self):
        self._update_motion_vel()
        self.motion_planning_interface.generate_motion_command()
        motion_commands = self.motion_planning_interface.get_motion_command()
        force_ff, torques, tau_ff_mpc, q_des, qd_des = self.mit_controller.step_run(self.controller_reset_buf, self.base_quat,
                                                                          self.base_ang_vel, self.base_lin_acc,
                                                                          self.dof_pos, self.dof_vel,
                                                                          self.feet_contact_state,
                                                                          motion_commands)
        self.motion_planning_interface.change_gait_planning(False)
        self.motion_planning_interface.change_body_planning(False)

        return force_ff, torques, tau_ff_mpc, q_des, qd_des

    def _cal_pd(self, tau_ff_mpc, q_des, qd_des, kp=25., kd=1.):
        v_max = 20.0233
        v_max /= 1.0
        tau_max = 33.5 * 1.0
        k = -3.953886

        kp_vec = torch.zeros_like(q_des)
        kd_vec = torch.zeros_like(qd_des)
        kp_vec[:] = kp
        kd_vec[:] = kd
        # kp_vec[:, [0, 3, 6, 9]] = 20.
        # kd_vec[:, [0, 3, 6, 9]] = 0.5
        torques = torch.clip(tau_ff_mpc + kp_vec * (q_des - self.dof_pos) + kd_vec * (qd_des - self.dof_vel), -tau_max, tau_max)
        tmp_max_torque = torch.clip(k * (self.dof_vel - v_max), 0, tau_max)
        tmp_min_torque = torch.clip(k * (self.dof_vel + v_max), -tau_max, 0)
        torques[:] = torch.where(self.dof_vel > tau_max / k + v_max,
                                 torch.clip(torques, -tau_max * torch.ones_like(torques), tmp_max_torque), torques)
        torques[:] = torch.where(self.dof_vel < -(tau_max / k + v_max),
                                 torch.clip(torques, tmp_min_torque, tau_max * torch.ones_like(torques)), torques)
        return torques

    def push_robots(self):
        self.root_states[:, 7:9] = torch_rand_float(-1., 1., (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = torch.tensor([0., 1.5], device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_pre_state(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_quat[:] = self.root_states[:, 3:7]
        self.euler_xyz[:] = get_euler_xyz2(self.base_quat)
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.feet_force[:] = self.contact_forces[:, self.feet_indices].view(self.num_envs, -1)
        self.feet_contact_state[:] = self.contact_forces[:, self.feet_indices, 2] > 2.0

        self.base_lin_acc[:] = quat_rotate_inverse(self.base_quat, ((self.root_states[:, 7:10] - self.last_base_lin_vel_rel_world) / self.sim_params.dt - self.gravity_acc))
        self.last_base_lin_vel_rel_world[:] = self.root_states[:, 7:10].clone().detach()

        self.controller_reset_buf[:] = 0

    def _create_sim(self):
        self.sim_params = gymapi.SimParams()
        self._config_sim_params()
        self._config_device()
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine,
                                       self.sim_params)

    def _config_sim_params(self):
        self.sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim_params.dt = self.sim_dt
        self.sim_params.substeps = SUB_STEPS
        self.sim_params.use_gpu_pipeline = USE_GPU
        self.sim_params.physx.contact_collection = gymapi.ContactCollection(2)
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = 14
        self.sim_params.physx.use_gpu = USE_GPU
        self.sim_params.physx.contact_offset = 0.01  # 0.02
        # self.sim_params.physx.friction_correlation_distance = 0.0025  # 0.025
        # self.sim_params.physx.friction_offset_threshold = 0.004  # 0.04
        self.sim_params.physx.rest_offset = 0.0  # 0.001
        self.sim_params.physx.bounce_threshold_velocity = 0.5
        self.sim_params.physx.max_depenetration_velocity = 1.0
        self.sim_params.physx.default_buffer_size_multiplier = 5.0
        self.sim_params.physx.max_gpu_contact_pairs = 8388608
        self.sim_params.physx.num_subscenes = 14

    def _config_device(self):
        self.sim_device = 'cuda:0'
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        if sim_device_type == 'cuda' and self.sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'
        self.graphics_device_id = self.sim_device_id
        if self.headless:
            self.graphics_device_id = -1

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.
        plane_params.dynamic_friction = 1.
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        asset_path = ASSERT_PATH
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = False
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 1000.
        asset_options.max_linear_velocity = 1000.
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)

        # get link name
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        base_name = BASE_NAME
        thigh_name = THIGH_NAME
        calf_name = CALF_NAME
        foot_name = FOOT_NAME
        thigh_names = [s for s in body_names if thigh_name in s]
        calf_names = [s for s in body_names if calf_name in s]
        feet_names = [s for s in body_names if foot_name in s]
        penalize_contacts_names = [base_name]
        penalize_contacts_names += thigh_names
        penalize_contacts_names += calf_names

        self.base_init_state = torch.tensor([0.0, 0.0, BODY_HEIGHT_INIT+0.002, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []

        rigid_shape_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for s in range(len(rigid_shape_props)):
            rigid_shape_props[s].friction = 1.0
        for j in range(self.num_dof):
            dof_props['driveMode'][j] = gymapi.DOF_MODE_EFFORT
            # dof_props['armature'][j] = 0.0

        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "a1", i, 1, 0)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)

            rigid_body_prop = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)
            # for bp in range(len(rigid_body_prop)):
            #     rigid_body_prop[bp].mass = 0.01
            # rigid_body_prop[0].mass = 12
            # rigid_body_prop[0].com = gymapi.Vec3(0.05, 0.0, 0.0)
            # rigid_body_prop[0].inertia.x = gymapi.Vec3(0.1, 0., 0.)
            # rigid_body_prop[0].inertia.y = gymapi.Vec3(0., 0.1, 0.)
            # rigid_body_prop[0].inertia.z = gymapi.Vec3(0., 0., 0.2)
            self.gym.set_actor_rigid_body_properties(env_handle, robot_handle, rigid_body_prop, recomputeInertia=False)
            body_prop = self.gym.get_actor_rigid_body_properties(env_handle, robot_handle)

            self.envs.append(env_handle)
            self.actor_handles.append(robot_handle)
        print("envs created!")

        # acquire link indices
        self.base_index = 0
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.penalize_contacts_indices = torch.zeros(len(penalize_contacts_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_name)
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], thigh_names[i])
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        for i in range(len(penalize_contacts_names)):
            self.penalize_contacts_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalize_contacts_names[i])

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = SPACING
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _init_buffers(self):
        self.common_step_counter = 0
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        actor_rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)  # shape: num_envs*num_bodies, xyz axis
        self.rigid_body_states = gymtorch.wrap_tensor(actor_rigid_body_state)

        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = self.net_contact_forces.view(self.num_envs, -1, 3)
        self.feet_force = self.contact_forces[:, self.feet_indices].view(self.num_envs, -1)
        self.feet_contact_state = torch.ones(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)  ### wsh_annotation: 1->contact
        self.rigid_body_states_reshape = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        self.base_quat = self.root_states[:, 3:7]

        default_dof_pos = DEFAULT_DOF_POS
        self.default_dof_pos = to_torch(default_dof_pos, device=self.device).repeat((self.num_envs, 1))

        self.gravity_vec = to_torch(get_axis_params(-1., 2), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

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
        self.base_lin_acc = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) - self.gravity_acc

        # mit controller reset flag
        self.controller_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        # commands
        self.vel_commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.gait_commands = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)  # period, duty, offset2, offset3, offset4, phase

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self._update_motion_gait(env_ids)

        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.euler_xyz[env_ids] = get_euler_xyz2(self.base_quat[env_ids])
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        self.base_lin_acc[env_ids] = -self.gravity_acc
        self.last_base_lin_vel_rel_world[env_ids] = self.root_states[env_ids, 7:10].clone().detach()

        self.feet_force[env_ids] = 0.0
        self.feet_contact_state[env_ids] = 0

        self.controller_reset_buf[env_ids] = 1
        self.progress_buf[env_ids] = 0

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos[env_ids]
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _resample_commands(self, env_ids):
        self.vel_commands[env_ids, 0] = 1.
        self.vel_commands[env_ids, 1] = 0.
        self.vel_commands[env_ids, 3] = 0.

        walk1 = [0.5, 0.75, 0.5, 0.25, 0.75, 0.]
        walk2 = [0.3, 0.75, 0.5, 0.25, 0.75, 0.]
        tort1 = [0.5, 0.5, 0.5, 0.5, 0., 0.]
        tort2 = [0.3, 0.5, 0.5, 0.5, 0., 0.]
        flying_tort1 = [0.5, 0.3, 0.5, 0.5, 0., 0.]
        flying_tort2 = [0.3, 0.3, 0.5, 0.5, 0., 0.]
        pace1 = [0.25, 0.6, 0.5, 0., 0.5, 0.]
        pace2 = [0.3, 0.5, 0.5, 0., 0.5, 0.]
        pronk1 = [0.3, 0.3, 0., 0., 0., 0.]
        pronk2 = [0.3, 0.5, 0., 0., 0., 0.]
        # bound1 = [0.3, 0.5, 0., 0.5, 0.5, 0.5]
        # bound2 = [0.3, 0.4, 0., 0.5, 0.5, 0.5]


        self.gait_commands[env_ids] = torch.tensor(tort2, dtype=torch.float, device=self.device, requires_grad=False)

    def _update_motion_gait(self, env_ids):
        gait_period_offset = (self.gait_commands[:, 0] - 0.5).unsqueeze(-1).repeat(1, 4)
        gait_duty_cycle_offset = (self.gait_commands[:, 1] - 0.5).unsqueeze(-1).repeat(1, 4)
        gait_phase_offset = self.gait_commands[:, 2:6].clone()
        gait_phase_offset[:, :2] = gait_phase_offset[:, :2].clone() - 0.5
        gait_to_change = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        gait_to_change[env_ids] = 1
        self.motion_planning_interface.update_gait_planning(gait_to_change, gait_period_offset, gait_duty_cycle_offset,
                                                            gait_phase_offset, None)

    def _update_motion_vel(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        self.vel_commands[:, 2] = torch.clip(0.8 * wrap_to_pi(self.vel_commands[:, 3] - heading), -1., 1.)
        # self.vel_commands[:, 2] = 1.
        vel_commands = self.vel_commands[:, :3].clone()
        self.motion_planning_interface.update_body_planning(True, None, None, None, None, vel_commands)

    def _set_viewer(self):
        self.enable_viewer_sync = True
        self.viewer = None
        self.viewer_pos_init = VIEWER_POS_INIT
        self.viewer_lookat_init = VIEWER_LOOKAT_INIT

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames")

            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_F, "free_cam")
            for i in range(9):
                self.gym.subscribe_viewer_keyboard_event(
                    self.viewer, getattr(gymapi, "KEY_" + str(i)), "lookat" + str(i))
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT_BRACKET, "prev_id")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT_BRACKET, "next_id")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_SPACE, "pause")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_W, "vx_plus")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "vx_minus")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_A, "left_turn")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_D, "right_turn")

            self.set_camera(self.viewer_pos_init, self.viewer_lookat_init)

        self.free_cam = False
        self.lookat_id = 0
        self.lookat_vec = torch.tensor([0, 2, 1], requires_grad=False, device=self.device)

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def lookat(self, i):
        look_at_pos = self.root_states[i, :3].clone()
        cam_pos = look_at_pos + self.lookat_vec
        self.set_camera(cam_pos, look_at_pos)

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            if not self.free_cam:
                self.lookat(self.lookat_id)
            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

                if not self.free_cam:
                    for i in range(9):
                        if evt.action == "lookat" + str(i) and evt.value > 0:
                            self.lookat(i)
                            self.lookat_id = i
                    if evt.action == "prev_id" and evt.value > 0:
                        self.lookat_id = (self.lookat_id - 1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "next_id" and evt.value > 0:
                        self.lookat_id = (self.lookat_id + 1) % self.num_envs
                        self.lookat(self.lookat_id)
                    if evt.action == "vx_plus" and evt.value > 0:
                        self.vel_commands[self.lookat_id, 0] += 0.2
                    if evt.action == "vx_minus" and evt.value > 0:
                        self.vel_commands[self.lookat_id, 0] -= 0.2
                    if evt.action == "left_turn" and evt.value > 0:
                        self.vel_commands[self.lookat_id, 3] += 0.5
                    if evt.action == "right_turn" and evt.value > 0:
                        self.vel_commands[self.lookat_id, 3] -= 0.5
                if evt.action == "free_cam" and evt.value > 0:
                    self.free_cam = not self.free_cam
                    if self.free_cam:
                        self.set_camera(self.viewer_pos_init, self.viewer_lookat_init)

                if evt.action == "pause" and evt.value > 0:
                    self.pause = True
                    while self.pause:
                        time.sleep(0.1)
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        for evt in self.gym.query_viewer_action_events(self.viewer):
                            if evt.action == "pause" and evt.value > 0:
                                self.pause = False
                        if self.gym.query_viewer_has_closed(self.viewer):
                            sys.exit()

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            self.gym.poll_viewer_events(self.viewer)
            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.dt * self.control_freq_inv  # render every control step
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()
            else:
                self.gym.poll_viewer_events(self.viewer)

            if not self.free_cam:
                p = self.gym.get_viewer_camera_transform(self.viewer, None).p
                cam_trans = torch.tensor([p.x, p.y, p.z], requires_grad=False, device=self.device)
                look_at_pos = self.root_states[self.lookat_id, :3].clone()
                self.lookat_vec = cam_trans - look_at_pos


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
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw], dim=1)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


class A1GaitEvaluate(A1Env):
    def __init__(self):
        self.num_envs = NUM_ENVS
        self.device = 'cuda:0'
        self._init_gait_buf()
        super().__init__()

        self._set_progress_bar()

    def _init_gait_buf(self):
        vx = torch.arange(-0.6, 3.1, 0.3, device=self.device)
        period = torch.arange(0.2, 0.76, 0.05, device=self.device)
        duty = torch.arange(0.2, 0.76, 0.05, device=self.device)
        phase2 = torch.arange(0, 0.95, 0.1, device=self.device)
        phase3 = torch.arange(0, 0.95, 0.1, device=self.device)
        phase4 = torch.arange(0, 0.95, 0.1, device=self.device)

        # vx = torch.arange(0.3, 0.5, 0.3, device=self.device)
        # period = torch.arange(0.3, 0.35, 0.1, device=self.device)
        # duty = torch.arange(0.5, 0.55, 0.1, device=self.device)
        # phase2 = torch.arange(0.5, 0.65, 0.1, device=self.device)
        # phase3 = torch.arange(0.5, 0.65, 0.1, device=self.device)
        # phase4 = torch.arange(0, 0.55, 0.1, device=self.device)

        command_table = permutations([vx, period, duty, phase2, phase3, phase4])
        # self.command_evaluate_table: [vx, period, duty, phase2, phase3, phase4] +
        # [vx_mean, vy_mean, vz_mean, wx_mean, wy_mean, wz_mean, rx_mean, ry_mean, rz_mean] +
        # [vx_std, vy_std, vz_std, wx_std, wy_std, wz_std, rx_std, ry_std, rz_std] +
        # [vx_bias_max, vy_bias_max, vz_bias_max, wx_bias_max, wy_bias_max, wz_bias_max, rx_bias_max, ry_bias_max, rz_bias_max] +
        # [h_mean, h_std, h_bias_max, target_position_bias_x, target_position_bias_y, power_mech_mean, power_total_mean, contact_force_max]
        self.command_evaluate_table = torch.cat([command_table, torch.zeros(command_table.shape[0], 35, device=self.device)], dim=1)
        self.command_evaluate_table[:, -1] = -1.  # -1 denotes it did not stick to the end
        self.total_command_num = len(self.command_evaluate_table)
        print(f"total num: {self.total_command_num}")
        self.table_index = torch.arange(self.total_command_num, dtype=torch.long, device=self.device, requires_grad=False)
        self.visited_count = 0
        self.visited_count_last = 0
        self.record_num = 1500  # during latest 3 second
        self.record_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        # self.record_command_buf: [vx, period, duty, phase2, phase3, phase4]
        self.record_command_buf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        # self.record_evaluate_buf: [vx, vy, vz, wx, wy, wz, rx, ry, rz, height, position_x, position_y, power_mech, power_total, contact_force_max]
        self.record_evaluate_buf = torch.zeros(self.num_envs, 15, self.record_num, dtype=torch.float, device=self.device, requires_grad=False)
        self.evaluate_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.stop_resample = False
        self.record_stop_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.save_interval = 100000
        self.create_save_path = True
        self.save_path = None

    def _resample_commands(self, env_ids):
        if self.stop_resample:
            self.record_stop_state[env_ids] = True
            if torch.all(self.record_stop_state):
                self.save_evaluate_table("final_" + str(self.visited_count))
                self.pbar.finish()
                sys.exit()
            self.record_index[env_ids] = -1
            self.record_command_buf[env_ids] = torch.tensor([0.0, 0.3, 0.5, 0.5, 0.0, 0.5], dtype=torch.float, device=self.device, requires_grad=False)
        else:
            if self.visited_count // (self.save_interval + 5000):
                self.save_evaluate_table("latest_" + str(self.visited_count))
                self.save_interval += 100000

            if self.visited_count > self.visited_count_last:
                self.pbar.update(self.visited_count + 1)
            self.visited_count_last = self.visited_count

            num = len(env_ids)
            remain_num = self.total_command_num - self.visited_count
            if num >= remain_num:
                self.stop_resample = True
                ids = env_ids[:remain_num]
                self.record_index[ids] = self.table_index[self.visited_count:]
                self.record_command_buf[ids] = self.command_evaluate_table[self.visited_count:, :6]
                self.visited_count = self.total_command_num
                if remain_num < num:
                    ids = env_ids[remain_num:]
                    self.record_stop_state[ids] = True
                    self.record_index[ids] = -1
                    self.record_command_buf[ids] = torch.tensor([0.0, 0.3, 0.5, 0.5, 0.0, 0.5], dtype=torch.float, device=self.device, requires_grad=False)
            else:
                self.visited_count += num
                self.record_index[env_ids] = self.table_index[self.visited_count_last:self.visited_count]
                self.record_command_buf[env_ids] = self.command_evaluate_table[self.visited_count_last:self.visited_count, :6]

        self.record_evaluate_buf[env_ids, :, :] = 0.0
        self.evaluate_index[env_ids] = 0
        self.vel_commands[env_ids, 0] = self.record_command_buf[env_ids, 0].clone()
        self.vel_commands[env_ids, 1] = 0.
        self.vel_commands[env_ids, 3] = 0.
        self.gait_commands[env_ids, :5] = self.record_command_buf[env_ids, 1:].clone()
        self.gait_commands[env_ids, 5] = 0.0

    def save_evaluate_table(self, file_name):
        if self.create_save_path:
            self.save_path = os.path.join('gait_evaluate_table', 'gait_evaluate_table' + '_{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.now()))
            os.makedirs(self.save_path, exist_ok=True)
            print(f"save path: {self.save_path}")
            self.create_save_path = False
        file_path = os.path.join(self.save_path, file_name+'.pt')
        torch.save(self.command_evaluate_table, file_path)

    def _update_pre_state(self):
        super()._update_pre_state()
        env_ids = (self.progress_buf >= 3500).nonzero(as_tuple=False).flatten()
        record_count_ids = self.progress_buf[env_ids] - 3500
        self.record_evaluate_buf[env_ids, :3, record_count_ids] = self.base_lin_vel[env_ids]
        self.record_evaluate_buf[env_ids, 3:6, record_count_ids] = self.base_ang_vel[env_ids]
        self.record_evaluate_buf[env_ids, 6:9, record_count_ids] = self.euler_xyz[env_ids]
        self.record_evaluate_buf[env_ids, 9, record_count_ids] = self.root_states[env_ids, 2]
        self.record_evaluate_buf[env_ids, 10:12, record_count_ids] = self.root_states[env_ids, :2]

        power_mech = self.torques[env_ids] * self.dof_vel[env_ids]
        power_total = power_mech.clone() + 0.26 * self.torques[env_ids] * self.torques[env_ids]
        power_mech = torch.clip(power_mech, min=0., max=None)
        power_total = torch.clip(power_total, min=0., max=None)
        self.record_evaluate_buf[env_ids, 12, record_count_ids] = torch.sum(power_mech, dim=-1)
        self.record_evaluate_buf[env_ids, 13, record_count_ids] = torch.sum(power_total, dim=-1)
        self.record_evaluate_buf[env_ids, 14, record_count_ids] = torch.max(self.feet_force[env_ids], dim=-1).values

        self.evaluate_index[env_ids] += 1

        # self.record_evaluate_buf[env_ids, 0] += self.root_states[env_ids, 2]
        # self.record_evaluate_buf[env_ids, 1] += self.base_lin_vel[env_ids, 0]
        # power_mech = self.torques[env_ids] * self.dof_vel[env_ids]
        # power_total = power_mech.clone() + 0.26 * self.torques[env_ids] * self.torques[env_ids]
        # power_mech = torch.clip(power_mech, min=0., max=None)
        # power_total = torch.clip(power_total, min=0., max=None)
        # self.record_evaluate_buf[env_ids, 2] += torch.sum(power_mech, dim=-1)
        # self.record_evaluate_buf[env_ids, 3] += torch.sum(power_total, dim=-1)

    def _set_progress_bar(self):
        widgets = ['Progress: ', progressbar.Percentage(), ' ', progressbar.Bar('#'), ' ', progressbar.Timer(), ' ',
                   progressbar.ETA(), ' ']
        self.pbar = progressbar.ProgressBar(widgets=widgets, maxval=self.total_command_num).start()

    def post_physics_step(self):
        self.progress_buf[:] += 1
        self.common_step_counter += 1

        self.check_termination()

        not_stopped = torch.where(self.record_stop_state, torch.zeros_like(self.record_stop_state), torch.ones_like(self.record_stop_state))
        env_ids = (self.timeout_buf & not_stopped).nonzero(as_tuple=False).flatten()
        # self.record_evaluate_buf[env_ids] *= self.sim_dt
        table_index = self.record_index[env_ids]
        self.command_evaluate_table[table_index, -35:-26] = torch.mean(self.record_evaluate_buf[env_ids, :9], dim=-1)
        self.command_evaluate_table[table_index, -26:-17] = torch.std(self.record_evaluate_buf[env_ids, :9], dim=-1)
        bias = self.record_evaluate_buf[env_ids, :9].clone()
        bias[:, 0, :] -= self.command_evaluate_table[table_index, 0].unsqueeze(-1)
        self.command_evaluate_table[table_index, -17:-8] = torch.max(torch.abs(bias), dim=-1).values
        self.command_evaluate_table[table_index, -8] = torch.mean(self.record_evaluate_buf[env_ids, 9], dim=-1)
        self.command_evaluate_table[table_index, -7] = torch.std(self.record_evaluate_buf[env_ids, 9], dim=-1)
        self.command_evaluate_table[table_index, -6] = torch.max(torch.abs(self.record_evaluate_buf[env_ids, 9] - 0.3), dim=-1).values
        target_x = self.env_origins[env_ids, 0] + self.command_evaluate_table[table_index, 0] * 10.0  # 10 seconds
        self.command_evaluate_table[table_index, -5] = self.record_evaluate_buf[env_ids, 10, -1] - target_x
        self.command_evaluate_table[table_index, -4] = self.record_evaluate_buf[env_ids, 11, -1]
        self.command_evaluate_table[table_index, -3] = torch.mean(self.record_evaluate_buf[env_ids, 12], dim=-1)
        self.command_evaluate_table[table_index, -2] = torch.mean(self.record_evaluate_buf[env_ids, 13], dim=-1)
        self.command_evaluate_table[table_index, -1] = torch.max(torch.abs(self.record_evaluate_buf[env_ids, 14]), dim=-1).values

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)


def permutations(tensor_list):
    grids = torch.meshgrid(tensor_list)
    result = torch.stack(grids, -1).view(-1, len(tensor_list))
    return result


if __name__ == "__main__":
    # a1_gait_evaluate = A1GaitEvaluate()
    # a1_gait_evaluate.run()
    a1 = A1Env()
    a1.run()
