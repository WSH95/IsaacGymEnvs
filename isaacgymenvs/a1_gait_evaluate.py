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
import progressbar

NUM_ENVS = 6
HEADLESS = False
USE_GPU = True
SIM_DT = 0.002
DECIMATION = 1
MAX_EPISODE_LEN_S = 10.0
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


class A1Env:
    def __init__(self):
        self.num_envs = NUM_ENVS
        self.headless = HEADLESS
        self.decimation = DECIMATION
        self.sim_dt = SIM_DT
        self.dt = self.sim_dt * self.decimation
        self.max_episode_length = int(MAX_EPISODE_LEN_S / self.dt + 0.5)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

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
        if self.common_step_counter % 10 == 0:
            self.render()

        for i in range(self.decimation):
            torques = self._cal_torque()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques[:] = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self._update_pre_state()

        self.post_physics_step()

    def post_physics_step(self):
        self.progress_buf[:] += 1
        self.common_step_counter += 1

        self.check_termination()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
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
        torques, tau_ff_mpc, q_des, qd_des = self.mit_controller.step_run(self.controller_reset_buf, self.base_quat,
                                                                          self.base_ang_vel, self.base_lin_acc,
                                                                          self.dof_pos, self.dof_vel,
                                                                          self.feet_contact_state,
                                                                          motion_commands)
        self.motion_planning_interface.change_gait_planning(False)
        self.motion_planning_interface.change_body_planning(False)

        return torques

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
            dof_props['armature'][j] = 0.01

        print("Creating env...")
        for i in tqdm(range(self.num_envs)):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*(pos + self.base_init_state[:3]))

            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            robot_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "a1", i, 1, 0)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)

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
        self.vel_commands[env_ids, 0] = 1.2
        self.vel_commands[env_ids, 1] = 0.
        self.vel_commands[env_ids, 3] = 0.

        self.gait_commands[env_ids] = torch.tensor([0.4, 0.4, 0.5, 0.6, 0.1, 0.0], dtype=torch.float, device=self.device, requires_grad=False)

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
        period = torch.arange(0.2, 0.75, 0.1, device=self.device)
        duty = torch.arange(0.2, 0.75, 0.1, device=self.device)
        phase2 = torch.arange(0, 0.95, 0.1, device=self.device)
        phase3 = torch.arange(0, 0.95, 0.1, device=self.device)
        phase4 = torch.arange(0, 0.95, 0.1, device=self.device)

        # vx = torch.arange(0., 0.2, 0.3, device=self.device)
        # period = torch.arange(0.3, 0.35, 0.1, device=self.device)
        # duty = torch.arange(0.5, 0.55, 0.1, device=self.device)
        # phase2 = torch.arange(0, 0.55, 0.1, device=self.device)
        # phase3 = torch.arange(0, 0.55, 0.1, device=self.device)
        # phase4 = torch.arange(0, 0.55, 0.1, device=self.device)

        command_table = permutations([vx, period, duty, phase2, phase3, phase4])
        # self.command_evaluate_table: [vx, period, duty, phase2, phase3, phase4, height_mean, vx_mean, mech_power_mean, total_power_mean]
        self.command_evaluate_table = torch.cat([command_table, torch.zeros(command_table.shape[0], 4, device=self.device)], dim=1)
        self.command_evaluate_table[:, -1] = -1.  # -1 denotes it did not stick to the end
        self.total_command_num = len(self.command_evaluate_table)
        self.table_index = torch.arange(self.total_command_num, dtype=torch.long, device=self.device, requires_grad=False)
        self.visited_count = 0
        self.record_num = 500  # during latest 1 second
        self.record_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        # self.record_command_buf: [vx, period, duty, phase2, phase3, phase4]
        self.record_command_buf = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False)
        # self.record_evaluate_buf: [height_buf, vx_buf, mech_power_buf, total_power_buf]
        self.record_evaluate_buf = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.evaluate_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.stop_resample = False
        self.record_stop_state = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.save_interval = 500

    def _resample_commands(self, env_ids):
        if self.stop_resample:
            self.record_stop_state[env_ids] = True
            if torch.all(self.record_stop_state):
                self.save_evaluate_table("final")
                self.pbar.finish()
                sys.exit()
            self.record_index[env_ids] = -1
            self.record_command_buf[env_ids] = torch.tensor([0.0, 0.3, 0.5, 0.5, 0.0, 0.5], dtype=torch.float, device=self.device, requires_grad=False)
        else:
            if self.visited_count // (self.save_interval + 5000):
                self.save_evaluate_table("latest_" + str(self.visited_count))
                self.save_interval += 500
                self.pbar.update(self.visited_count + 1)

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
                visited_count_last = self.visited_count
                self.visited_count += num
                self.record_index[env_ids] = self.table_index[visited_count_last:self.visited_count]
                self.record_command_buf[env_ids] = self.command_evaluate_table[visited_count_last:self.visited_count, :6]

        self.record_evaluate_buf[env_ids] = 0.0
        self.evaluate_index[env_ids] = 0
        self.vel_commands[env_ids, 0] = self.record_command_buf[env_ids, 0].clone()
        self.vel_commands[env_ids, 1] = 0.
        self.vel_commands[env_ids, 3] = 0.
        self.gait_commands[env_ids, :5] = self.record_command_buf[env_ids, 1:].clone()
        self.gait_commands[env_ids, 5] = 0.0

    def save_evaluate_table(self, file_name):
        os.makedirs("gait_evaluate_table2", exist_ok=True)
        torch.save(self.command_evaluate_table, "gait_evaluate_table2/"+file_name+".pt")

    def _update_pre_state(self):
        super()._update_pre_state()
        env_ids = (self.progress_buf >= 4500).nonzero(as_tuple=False).flatten()
        self.record_evaluate_buf[env_ids, 0] += self.root_states[env_ids, 2]
        self.record_evaluate_buf[env_ids, 1] += self.base_lin_vel[env_ids, 0]
        power_mech = self.torques[env_ids] * self.dof_vel[env_ids]
        power_total = power_mech.clone() + 0.26 * self.torques[env_ids] * self.torques[env_ids]
        power_mech = torch.clip(power_mech, min=0., max=None)
        power_total = torch.clip(power_total, min=0., max=None)
        self.record_evaluate_buf[env_ids, 2] += torch.sum(power_mech, dim=-1)
        self.record_evaluate_buf[env_ids, 3] += torch.sum(power_total, dim=-1)

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
        self.record_evaluate_buf[env_ids] *= self.sim_dt
        table_index = self.record_index[env_ids]
        self.command_evaluate_table[table_index, -4:] = self.record_evaluate_buf[env_ids].clone()

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
