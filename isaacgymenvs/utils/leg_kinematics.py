# -*- coding: utf-8 -*-
# Created by Shuhan Wang on 2024/3/27.
#

import torch


# This class is used to calculate the kinematics and velocity jacobian of four quadruped leg. The order of the leg is: front left, front right, back left, back right
class QuadrupedLegKinematics:
    def __init__(self, hip_length, thigh_length, calf_length, device='cuda:0', side_sign=[1, -1, 1, -1]):
        self.l1 = hip_length
        self.l2 = thigh_length
        self.l3 = calf_length
        self.num_robots = None

        # if the type of hip_length, thigh_length, calf_length is torch.Tensor, assert the length of the three parameters is the same.
        if isinstance(hip_length, torch.Tensor) and isinstance(thigh_length, torch.Tensor) and isinstance(calf_length,
                                                                                                          torch.Tensor):
            assert hip_length.shape == thigh_length.shape == calf_length.shape, "The shape of hip_length, thigh_length, calf_length should be the same."
            self.num_robots = hip_length.shape[0]
        else:
            assert isinstance(hip_length, (int, float)), "The type of hip_length should be int or float."
            assert isinstance(thigh_length, (int, float)), "The type of thigh_length should be int or float."
            assert isinstance(calf_length, (int, float)), "The type of calf_length should be int or float."

        self.device = device
        self.side_sign = torch.tensor(side_sign, dtype=torch.float32, device=device).reshape(1, 4, 1)

    def forward_kinematics(self, q: torch.Tensor):
        """
        Forward kinematics of four quadruped leg.
        :param q: The joint angles of four quadruped leg, the data type is torch Tensor, the shape is (num_robots, 4, 3), the order of the leg is: front left, front right, back left, back right.
        :return1: The position of the end of the calf, the data type is torch Tensor, the shape is (num_robots, 4, 3), the order of the leg is: front left, front right, back left, back right.
        :return2: The jocobbis of the end of the calf, the data type is torch Tensor, the shape is (num_robots, 4, 3, 3), the order of the leg is: front left, front right, back left, back right.
        :return3: The inverse jocobbis of the end of the calf, the data type is torch Tensor, the shape is (num_robots, 4, 3, 3), the order of the leg is: front left, front right, back left, back right.
        """
        num_robots = q.shape[0]
        if self.num_robots is not None:
            assert num_robots == self.num_robots, "The number of robots should be the same as the number of hip_length, thigh_length, calf_length."
            l1 = self.l1.clone()
            l2 = self.l2.clone()
            l3 = self.l3.clone()
        else:
            l1 = torch.ones((num_robots, 4), dtype=torch.float32, device=self.device) * self.l1
            l2 = torch.ones((num_robots, 4), dtype=torch.float32, device=self.device) * self.l2
            l3 = torch.ones((num_robots, 4), dtype=torch.float32, device=self.device) * self.l3
        side_sign = self.side_sign.repeat(num_robots, 1, 1).squeeze(-1)
        position = torch.zeros((num_robots, 4, 3), dtype=torch.float32, device=self.device)
        jacobian = torch.zeros((num_robots, 4, 3, 3), dtype=torch.float32, device=self.device)
        inverse_jacobian = torch.zeros((num_robots, 4, 3, 3), dtype=torch.float32, device=self.device)

        s = torch.sin(q)
        c = torch.cos(q)
        s1 = s[..., 0]
        s2 = s[..., 1]
        s3 = s[..., 2]
        c1 = c[..., 0]
        c2 = c[..., 1]
        c3 = c[..., 2]
        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        position[..., 0] = -l2 * s2 - l3 * s23
        position[..., 1] = side_sign * l1 * c1 + l2 * s1 * c2 + l3 * s1 * c23
        position[..., 2] = side_sign * l1 * s1 - l2 * c1 * c2 - l3 * c1 * c23

        jacobian[..., 0, 0] = 0
        jacobian[..., 0, 1] = -l2 * c2 - l3 * c23
        jacobian[..., 0, 2] = -l3 * c23
        jacobian[..., 1, 0] = side_sign * l1 * (-s1) + l2 * c1 * c2 + l3 * c1 * c23
        jacobian[..., 1, 1] = -l2 * s1 * s2 - l3 * s1 * s23
        jacobian[..., 1, 2] = -l3 * s1 * s23
        jacobian[..., 2, 0] = side_sign * l1 * c1 + l2 * s1 * c2 + l3 * s1 * c23
        jacobian[..., 2, 1] = l2 * c1 * s2 + l3 * c1 * s23
        jacobian[..., 2, 2] = l3 * c1 * s23

        # for n in range(num_robots):
        #     for i in range(4):
        #         inverse_jacobian[n, i] = torch.inverse(jacobian[n, i])
        inverse_jacobian[:] = torch.inverse(jacobian)

        return position, jacobian, inverse_jacobian

    def inverse_kinematics(self, p: torch.Tensor):
        """
        Inverse kinematics of four quadruped leg.
        :param p: The position of the end of the calf, the data type is torch Tensor, the shape is (num_robots, 4, 3), the order of the leg is: front left, front right, back left, back right.
        :return: The joint angles of four quadruped leg, the data type is torch Tensor, the shape is (num_robots, 4, 3), the order of the leg is: front left, front right, back left, back right.
        """
        num_robots = p.shape[0]
        if self.num_robots is not None:
            assert num_robots == self.num_robots, "The number of robots should be the same as the number of hip_length, thigh_length, calf_length."
            l1 = self.l1.clone()
            l2 = self.l2.clone()
            l3 = self.l3.clone()
        else:
            l1 = torch.ones((num_robots, 4), dtype=torch.float32, device=self.device) * self.l1
            l2 = torch.ones((num_robots, 4), dtype=torch.float32, device=self.device) * self.l2
            l3 = torch.ones((num_robots, 4), dtype=torch.float32, device=self.device) * self.l3

        side_sign = self.side_sign.repeat(num_robots, 1, 1)
        q = torch.zeros((num_robots, 4, 3), dtype=torch.float32, device=self.device)

        p_right_leg_trans = p.clone()
        p_right_leg_trans[..., :2] *= side_sign
        p_x = p_right_leg_trans[..., 0]
        p_y = p_right_leg_trans[..., 1]
        p_z = p_right_leg_trans[..., 2]

        d_square_origin2projectedPointYZ = torch.sum(torch.square(p_right_leg_trans[..., 1:]), dim=-1)
        d_tanPoint2projectedPointYZ = torch.sqrt(d_square_origin2projectedPointYZ - torch.square(l1))

        # cancel l1 in numerator and 'distance2_origin2projectedPointYZ' in denominator.
        y_tanPoint_normed = l1 * p_y - p_z * d_tanPoint2projectedPointYZ
        z_tanPoint_normed = l1 * p_z + p_y * d_tanPoint2projectedPointYZ
        q[..., 0] = torch.atan2(z_tanPoint_normed, y_tanPoint_normed)

        xSquare_add_zSquare = torch.square(p_x) + torch.square(d_tanPoint2projectedPointYZ)
        beata = torch.acos((torch.square(l2) + torch.square(l3) - xSquare_add_zSquare) / (2 * l2 * l3))
        alpha = torch.acos(
            (torch.square(l2) + xSquare_add_zSquare - torch.square(l3)) / (2 * l2 * torch.sqrt(xSquare_add_zSquare)))
        gamma = torch.atan2(-p_x, d_tanPoint2projectedPointYZ)

        q[..., 1] = gamma + side_sign.squeeze(-1) * alpha
        q[..., 2] = (beata - torch.pi) * side_sign.squeeze(-1)
        wrap_to_pi(q)
        q *= side_sign

        return q


class QuadrupedLegKinematics2:
    def __init__(self, num_robots, hip_length, thigh_length, calf_length, device='cuda:0', side_sign=[1, -1, 1, -1]):
        self.num_robots = num_robots
        self.device = device
        self.l1 = torch.ones((num_robots, 4), dtype=torch.float32, device=self.device) * hip_length
        self.l2 = torch.ones((num_robots, 4), dtype=torch.float32, device=self.device) * thigh_length
        self.l3 = torch.ones((num_robots, 4), dtype=torch.float32, device=self.device) * calf_length

        self.side_sign = torch.tensor(side_sign, dtype=torch.float32, device=device).reshape(1, 4, 1).repeat(num_robots,
                                                                                                             1, 1)
        self.side_sign_squeeze = self.side_sign.squeeze(-1)

        # for forward kinematics
        self.position = torch.zeros((num_robots, 4, 3), dtype=torch.float32, device=self.device)
        self.jacobian = torch.zeros((num_robots, 4, 3, 3), dtype=torch.float32, device=self.device)
        self.inverse_jacobian = torch.zeros((num_robots, 4, 3, 3), dtype=torch.float32, device=self.device)
        self.sin_q = torch.zeros((num_robots, 4, 3), dtype=torch.float32, device=self.device)
        self.cos_q = torch.zeros((num_robots, 4, 3), dtype=torch.float32, device=self.device)
        self.s1 = self.sin_q.view(num_robots, 4, 3)[..., 0]
        self.s2 = self.sin_q.view(num_robots, 4, 3)[..., 1]
        self.s3 = self.sin_q.view(num_robots, 4, 3)[..., 2]
        self.c1 = self.cos_q.view(num_robots, 4, 3)[..., 0]
        self.c2 = self.cos_q.view(num_robots, 4, 3)[..., 1]
        self.c3 = self.cos_q.view(num_robots, 4, 3)[..., 2]
        self.c23 = torch.zeros_like(self.s1)
        self.s23 = torch.zeros_like(self.s1)

        # for inverse kinematics
        self.inverse_kin_q = torch.zeros((num_robots, 4, 3), dtype=torch.float32, device=self.device)
        self.p_right_leg_trans = torch.zeros((num_robots, 4, 3), dtype=torch.float32, device=self.device)
        self.p_x = self.p_right_leg_trans.view(num_robots, 4, 3)[..., 0]
        self.p_y = self.p_right_leg_trans.view(num_robots, 4, 3)[..., 1]
        self.p_z = self.p_right_leg_trans.view(num_robots, 4, 3)[..., 2]
        self.d_square_origin2projectedPointYZ = torch.zeros((num_robots, 4), dtype=torch.float32, device=self.device)
        self.d_tanPoint2projectedPointYZ = torch.zeros((num_robots, 4), dtype=torch.float32, device=self.device)
        self.y_tanPoint_normed = torch.zeros((num_robots, 4), dtype=torch.float32, device=self.device)
        self.z_tanPoint_normed = torch.zeros((num_robots, 4), dtype=torch.float32, device=self.device)
        self.xSquare_add_zSquare = torch.zeros((num_robots, 4), dtype=torch.float32, device=self.device)
        self.beata = torch.zeros((num_robots, 4), dtype=torch.float32, device=self.device)
        self.alpha = torch.zeros((num_robots, 4), dtype=torch.float32, device=self.device)
        self.gamma = torch.zeros((num_robots, 4), dtype=torch.float32, device=self.device)

    def forward_kinematics(self, q: torch.Tensor):
        self.sin_q[:] = torch.sin(q)
        self.cos_q[:] = torch.cos(q)

        self.c23[:] = self.c2 * self.c3 - self.s2 * self.s3
        self.s23[:] = self.s2 * self.c3 + self.c2 * self.s3

        self.position[..., 0] = -self.l2 * self.s2 - self.l3 * self.s23
        self.position[
            ..., 1] = self.side_sign_squeeze * self.l1 * self.c1 + self.l2 * self.s1 * self.c2 + self.l3 * self.s1 * self.c23
        self.position[
            ..., 2] = self.side_sign_squeeze * self.l1 * self.s1 - self.l2 * self.c1 * self.c2 - self.l3 * self.c1 * self.c23

        self.jacobian[..., 0, 0] = 0
        self.jacobian[..., 0, 1] = -self.l2 * self.c2 - self.l3 * self.c23
        self.jacobian[..., 0, 2] = -self.l3 * self.c23
        self.jacobian[..., 1, 0] = self.side_sign_squeeze * self.l1 * (
            -self.s1) + self.l2 * self.c1 * self.c2 + self.l3 * self.c1 * self.c23
        self.jacobian[..., 1, 1] = -self.l2 * self.s1 * self.s2 - self.l3 * self.s1 * self.s23
        self.jacobian[..., 1, 2] = -self.l3 * self.s1 * self.s23
        self.jacobian[
            ..., 2, 0] = self.side_sign_squeeze * self.l1 * self.c1 + self.l2 * self.s1 * self.c2 + self.l3 * self.s1 * self.c23
        self.jacobian[..., 2, 1] = self.l2 * self.c1 * self.s2 + self.l3 * self.c1 * self.s23
        self.jacobian[..., 2, 2] = self.l3 * self.c1 * self.s23

        # for n in range(num_robots):
        #     for i in range(4):
        #         inverse_jacobian[n, i] = torch.inverse(jacobian[n, i])

        self.inverse_jacobian[:] = torch.inverse(self.jacobian)

        return self.position, self.jacobian, self.inverse_jacobian

    def inverse_kinematics(self, p: torch.Tensor):
        self.p_right_leg_trans[:] = p.clone()
        self.p_right_leg_trans[..., :2] *= self.side_sign

        self.d_square_origin2projectedPointYZ[:] = torch.sum(torch.square(self.p_right_leg_trans[..., 1:]), dim=-1)
        self.d_tanPoint2projectedPointYZ[:] = torch.sqrt(self.d_square_origin2projectedPointYZ - torch.square(self.l1))

        # cancel l1 in numerator and 'distance2_origin2projectedPointYZ' in denominator.
        self.y_tanPoint_normed[:] = self.l1 * self.p_y - self.p_z * self.d_tanPoint2projectedPointYZ
        self.z_tanPoint_normed[:] = self.l1 * self.p_z + self.p_y * self.d_tanPoint2projectedPointYZ
        self.inverse_kin_q[..., 0] = torch.atan2(self.z_tanPoint_normed, self.y_tanPoint_normed)

        self.xSquare_add_zSquare[:] = torch.square(self.p_x) + torch.square(self.d_tanPoint2projectedPointYZ)
        self.beata[:] = torch.acos(
            (torch.square(self.l2) + torch.square(self.l3) - self.xSquare_add_zSquare) / (2 * self.l2 * self.l3))
        self.alpha[:] = torch.acos((torch.square(self.l2) + self.xSquare_add_zSquare - torch.square(self.l3)) / (
                    2 * self.l2 * torch.sqrt(self.xSquare_add_zSquare)))
        self.gamma[:] = torch.atan2(-self.p_x, self.d_tanPoint2projectedPointYZ)

        self.inverse_kin_q[..., 1] = self.gamma + self.side_sign.squeeze(-1) * self.alpha
        self.inverse_kin_q[..., 2] = (self.beata - torch.pi) * self.side_sign.squeeze(-1)
        wrap_to_pi(self.inverse_kin_q)
        self.inverse_kin_q[:] *= self.side_sign

        return self.inverse_kin_q


def wrap_to_pi(angles: torch.Tensor):
    """
    Wrap the angles to [-pi, pi].
    :param angles: The angles to be wrapped, the data type is torch Tensor.
    :return: The wrapped angles, the data type is torch Tensor.
    """
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi


if __name__ == '__main__':
    hip_length = torch.tensor([0.0838], dtype=torch.float32, device='cuda:0').unsqueeze(0).repeat(3, 4)
    thigh_length = torch.tensor([0.2], dtype=torch.float32, device='cuda:0').unsqueeze(0).repeat(3, 4)
    calf_length = torch.tensor([0.2], dtype=torch.float32, device='cuda:0').unsqueeze(0).repeat(3, 4)
    quad = QuadrupedLegKinematics(hip_length, thigh_length, calf_length)
    # quad = QuadrupedLegKinematics2(1, 0.0838, 0.2, 0.2)
    cal_p = torch.zeros((3, 4, 3), dtype=torch.float32, device='cuda:0')
    cal_dp = torch.zeros_like(cal_p)
    cal_dq = torch.zeros_like(cal_p)
    cal_q = torch.zeros_like(cal_p)
    cal_jacobian = torch.zeros((3, 4, 3, 3), dtype=torch.float32, device='cuda:0')
    cal_inverse_jacobian = torch.zeros_like(cal_jacobian)
    # print(quad.side_sign[1])
    q = torch.tensor([
        [[0.0, 0.7954, -1.5908], [-0.0, 0.7954, -1.5908], [0.0, 0.7954, -1.5908], [-0.0, 0.7954, -1.5908]],
        [[0.0154, 0.0163, -1.0815], [-0.0182, 0.6052, -1.3785], [-0.1310, 0.8262, -1.4713], [0.2055, 0.1864, -1.0612]],
        [[-0.09, 1.16, -1.206], [-0.145, 0.148, -1.194], [-0.138, 0.267, -1.34], [0.138, 1.168, -1.123]]
    ], dtype=torch.float32, device='cuda:0')
    dq = torch.tensor([
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[8.0143e-03, 5.6418e+00, -8.3873e+00], [9.3160e-01, -5.6518e+00, -8.8629e+00],
         [4.9887e-01, -4.8558e+00, -1.4429e+01], [7.4150e-02, 5.2076e+00, -3.0184e+00]],
        [[0.348, 1.636, -5.89], [-0.221, 6.078, -3.982], [1.029, 4.885, -2.667], [0.653, 2.241, -6.264]]
    ], dtype=torch.float32, device='cuda:0')
    p = torch.tensor([
        [[0., 0.0838, -0.28], [0., -0.0838, -0.28], [0., 0.0838, -0.28], [0., -0.0838, -0.28]],
        [[0.1717, 0.0884, -0.2955], [0.0259, -0.0894, -0.3060], [-0.0268, 0.0445, -0.3038], [0.1164, -0.0158, -0.3350]],
        [[-0.174, 0.058, -0.286], [0.144, -0.126, -0.283], [0.123, 0.043, -0.297], [-0.193, -0.045, -0.287]]
    ], dtype=torch.float32, device='cuda:0')
    dp = torch.tensor([
        [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        [[-0.8623, -0.0053, 0.4994], [3.0069, 0.3103, 1.3012], [3.7400, 0.3612, 1.6135], [-1.3042, 0.0540, -0.1412]],
        [[0.719, 0.13, 0.357], [-1.412, -0.089, -0.154], [-1.154, 0.288, -0.086], [0.628, 0.136, 0.343]]
    ], dtype=torch.float32, device='cuda:0')

    # for i in range(3):
    #     cal_p[i], cal_jacobian[i], cal_inverse_jacobian[i] = quad.forward_kinematics(q[i].unsqueeze(0))
    #     cal_dp[i] = torch.matmul(cal_jacobian[i], dq[i].unsqueeze(-1)).squeeze(-1)
    #     cal_dq[i] = torch.matmul(cal_inverse_jacobian[i], dp[i].unsqueeze(-1)).squeeze(-1)
    #     cal_q[i] = quad.inverse_kinematics(p[i].unsqueeze(0))

    cal_p, cal_jacobian, cal_inverse_jacobian = quad.forward_kinematics(q)
    cal_dp = torch.matmul(cal_jacobian, dq.unsqueeze(-1)).squeeze(-1)
    cal_dq = torch.matmul(cal_inverse_jacobian, dp.unsqueeze(-1)).squeeze(-1)
    cal_q = quad.inverse_kinematics(p)

    print(f"cal_jacobian: {cal_jacobian}")
    print(f"cal_inverse_jacobian: {cal_inverse_jacobian}")
    print(f"cal_q: {cal_q}")
    print(f"cal_dq: {cal_dq}")
    print(f"cal_p: {cal_p}")
    print(f"cal_dp: {cal_dp}")
