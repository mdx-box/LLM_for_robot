"""
Adapted from OmniGibson and the Lula IK solver
"""
import omnigibson.lazy as lazy
import numpy as np

class IKSolver:
    """
    Class for thinly wrapping Lula IK solver
    """

    def __init__(
        self,
        robot_description_path, #机器人描述文件地址
        robot_urdf_path, #机器人urdf文件地址
        eef_name, #末端执行器名称
        reset_joint_pos, #机器人重置关节位置
        world2robot_homo, #世界坐标系到机器人坐标系的齐次变换矩阵
    ):
        # Create robot description, kinematics, and config
        #使用 lazy.lula.load_robot 函数加载机器人描述和 URDF 文件。
        self.robot_description = lazy.lula.load_robot(robot_description_path, robot_urdf_path)
        #机器人描述中创建运动学模型。
        self.kinematics = self.robot_description.kinematics()
        #创建一个 CyclicCoordDescentIkConfig 对象，用于配置 IK 求解器
        self.config = lazy.lula.CyclicCoordDescentIkConfig()
        self.eef_name = eef_name
        self.reset_joint_pos = reset_joint_pos
        self.world2robot_homo = world2robot_homo

    def solve(
        self,
        target_pose_homo, #目标位姿在世界坐标系中的齐次变换矩阵
        position_tolerance=0.01,
        orientation_tolerance=0.05,
        position_weight=1.0,
        orientation_weight=0.05,
        max_iterations=150,
        initial_joint_pos=None,
    ):
        #用于计算机器人的逆运动学（IK）解，即给定一个目标位姿（位置和方向），计算出机器人需要的关节位置，使得末端执行器能够到达这个目标位姿。
        """
        Backs out joint positions to achieve desired @target_pos and @target_quat

        Args:
            target_pose_homo (np.ndarray): [4, 4] homogeneous transformation matrix of the target pose in world frame
            position_tolerance (float): Maximum position error (L2-norm) for a successful IK solution
            orientation_tolerance (float): Maximum orientation error (per-axis L2-norm) for a successful IK solution
            position_weight (float): Weight for the relative importance of position error during CCD
            orientation_weight (float): Weight for the relative importance of position error during CCD
            max_iterations (int): Number of iterations used for each cyclic coordinate descent.
            initial_joint_pos (None or n-array): If specified, will set the initial cspace seed when solving for joint
                positions. Otherwise, will use self.reset_joint_pos

        Returns:
            ik_results (lazy.lula.CyclicCoordDescentIkResult): IK result object containing the joint positions and other information.
        """
        # convert target pose to robot base frame
        #目标位姿从世界坐标系转换到机器人坐标系
        target_pose_robot = np.dot(self.world2robot_homo, target_pose_homo)
        target_pose_pos = target_pose_robot[:3, 3]
        target_pose_rot = target_pose_robot[:3, :3]
        #这行代码创建一个 Pose3 对象，用于表示 IK 求解的目标位姿。
        ik_target_pose = lazy.lula.Pose3(lazy.lula.Rotation3(target_pose_rot), target_pose_pos)
        # Set the cspace seed and tolerance
        initial_joint_pos = self.reset_joint_pos if initial_joint_pos is None else np.array(initial_joint_pos)
        #设置 IK 求解的初始关节位置、位置和方向误差的容忍度、权重以及最大迭代次数
        self.config.cspace_seeds = [initial_joint_pos]
        self.config.position_tolerance = position_tolerance
        self.config.orientation_tolerance = orientation_tolerance
        self.config.ccd_position_weight = position_weight
        self.config.ccd_orientation_weight = orientation_weight
        self.config.max_num_descents = max_iterations
        # Compute target joint positions
        #使用 compute_ik_ccd 函数计算 IK 解。
        ik_results = lazy.lula.compute_ik_ccd(self.kinematics, ik_target_pose, self.eef_name, self.config)
        return ik_results