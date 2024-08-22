# -*- coding: utf-8 -*-
# Created by Shuhan Wang on 2024/5/28.
#

# Data description
data_description0 = {
    'dim_1': 'Sample index (quadruped fault-tolerant samples)',
    'dim_2': 'Timestep (starting from 0, interval 0.02s)',
    'dim_3': [
        'base_pose_x', 'base_pose_y', 'base_pose_z',  # base_pose (xyz)
        'base_quat_x', 'base_quat_y', 'base_quat_z', 'base_quat_w',  # base_quat (xyzw)
        'base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z',  # base_lin_vel (xyz)
        'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z',  # base_ang_vel (xyz)
        'dof_pos_FL0', 'dof_pos_FL1', 'dof_pos_FL2', 'dof_pos_FR0', 'dof_pos_FR1', 'dof_pos_FR2',
        'dof_pos_HL0', 'dof_pos_HL1', 'dof_pos_HL2', 'dof_pos_HR0', 'dof_pos_HR1', 'dof_pos_HR2',  # dof_pos (12 joints)
        'dof_vel_FL0', 'dof_vel_FL1', 'dof_vel_FL2', 'dof_vel_FR0', 'dof_vel_FR1', 'dof_vel_FR2',
        'dof_vel_HL0', 'dof_vel_HL1', 'dof_vel_HL2', 'dof_vel_HR0', 'dof_vel_HR1', 'dof_vel_HR2',  # dof_vel (12 joints)
        'feet_pos_world_FL_x', 'feet_pos_world_FL_y', 'feet_pos_world_FL_z',
        'feet_pos_world_FR_x', 'feet_pos_world_FR_y', 'feet_pos_world_FR_z',
        'feet_pos_world_HL_x', 'feet_pos_world_HL_y', 'feet_pos_world_HL_z',
        'feet_pos_world_HR_x', 'feet_pos_world_HR_y', 'feet_pos_world_HR_z',  # feet_position_world (4*xyz)
        'feet_lin_vel_world_FL_x', 'feet_lin_vel_world_FL_y', 'feet_lin_vel_world_FL_z',
        'feet_lin_vel_world_FR_x', 'feet_lin_vel_world_FR_y', 'feet_lin_vel_world_FR_z',
        'feet_lin_vel_world_HL_x', 'feet_lin_vel_world_HL_y', 'feet_lin_vel_world_HL_z',
        'feet_lin_vel_world_HR_x', 'feet_lin_vel_world_HR_y', 'feet_lin_vel_world_HR_z',  # feet_lin_vel_world (4*xyz)
        'feet_pos_body_FL_x', 'feet_pos_body_FL_y', 'feet_pos_body_FL_z',
        'feet_pos_body_FR_x', 'feet_pos_body_FR_y', 'feet_pos_body_FR_z',
        'feet_pos_body_HL_x', 'feet_pos_body_HL_y', 'feet_pos_body_HL_z',
        'feet_pos_body_HR_x', 'feet_pos_body_HR_y', 'feet_pos_body_HR_z',  # feet_position_body (4*xyz)
        'feet_lin_vel_body_FL_x', 'feet_lin_vel_body_FL_y', 'feet_lin_vel_body_FL_z',
        'feet_lin_vel_body_FR_x', 'feet_lin_vel_body_FR_y', 'feet_lin_vel_body_FR_z',
        'feet_lin_vel_body_HL_x', 'feet_lin_vel_body_HL_y', 'feet_lin_vel_body_HL_z',
        'feet_lin_vel_body_HR_x', 'feet_lin_vel_body_HR_y', 'feet_lin_vel_body_HR_z',  # feet_lin_vel_body (4*xyz)
        'feet_force_FL_x', 'feet_force_FL_y', 'feet_force_FL_z',
        'feet_force_FR_x', 'feet_force_FR_y', 'feet_force_FR_z',
        'feet_force_HL_x', 'feet_force_HL_y', 'feet_force_HL_z',
        'feet_force_HR_x', 'feet_force_HR_y', 'feet_force_HR_z',  # feet_force (4*xyz)
        'feet_contact_FL', 'feet_contact_FR', 'feet_contact_HL', 'feet_contact_HR',  # feet_contact_state (4)
        'command_vx', 'command_vy', 'command_omega',  # commands (vx, vy, omega)
        'torque_FL0', 'torque_FL1', 'torque_FL2', 'torque_FR0', 'torque_FR1', 'torque_FR2',
        'torque_HL0', 'torque_HL1', 'torque_HL2', 'torque_HR0', 'torque_HR1', 'torque_HR2',  # torques (12 joints)
        'last_action_FL0', 'last_action_FL1', 'last_action_FL2', 'last_action_FR0', 'last_action_FR1', 'last_action_FR2',
        'last_action_HL0', 'last_action_HL1', 'last_action_HL2', 'last_action_HR0', 'last_action_HR1', 'last_action_HR2',  # last_actions_raw (12 joints)
        'motor_broken_state',  # motor_broken_table (0-11, or -1 for no motor broken)
        'reset_state',  # reset_buf (0 or 1)
        'push_velocity_x', 'push_velocity_y',  # current_push_velocity (xy)
        'step_height',  # env_step_height
        'init_position_bias_rel_world_x', 'init_position_bias_rel_world_y', 'init_position_bias_rel_world_z',  # init_position_bias_rel_world (xyz)
        'feet_height_rel_ground_FL', 'feet_height_rel_ground_FR', 'feet_height_rel_ground_HL', 'feet_height_rel_ground_HR',  # feet_height_rel_ground (4)
    ],
    'notes': '10s total. simulation-adaptive: set motor fault at 5s. simulation-reliable: step located at 2m in front of the robot.'
}