import time
import numpy as np
import os
import datetime
import transform_utils as T
import trimesh
import open3d as o3d
import imageio
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.usd_utils import PoseAPI, mesh_prim_mesh_to_trimesh_mesh, mesh_prim_shape_to_trimesh_mesh
from omnigibson.robots.fetch import Fetch
from omnigibson.controllers import IsGraspingState
from og_utils import OGCamera
import torch
from utils import (
    bcolors,
    get_clock_time,
    angle_between_rotmat,
    angle_between_quats,
    get_linear_interpolation_steps,
    linear_interpolate_poses,
)
from omnigibson.robots.manipulation_robot import ManipulationRobot
from omnigibson.controllers.controller_base import ControlType, BaseController

# Don't use GPU dynamics and use flatcache for performance boost
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = False

# some customization to the OG functions
def custom_clip_control(self, control):
    """
    Clips the inputted @control signal based on @control_limits.

    Args:
        control (Array[float]): control signal to clip

    Returns:
        Array[float]: Clipped control signal
    """
    clipped_control = control.clip(
        self._control_limits[self.control_type][0][self.dof_idx],
        self._control_limits[self.control_type][1][self.dof_idx],
    )
    idx = (
        self._dof_has_limits[self.dof_idx]
        if self.control_type == ControlType.POSITION
        else [True] * self.control_dim
    )
    if len(control) > 1:
        control[idx] = clipped_control[idx]
    return control

Fetch._initialize = ManipulationRobot._initialize
BaseController.clip_control = custom_clip_control

class ReKepOGEnv:
    def __init__(self, config, scene_file, verbose=False):
        #定义一些初始值
        self.video_cache = []
        self.config = config
        self.verbose = verbose
        self.config['scene']['scene_file'] = scene_file
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']
        # create omnigibson environment
        self.step_counter = 0
        #这里定义一个基本的环境
        self.og_env = og.Environment(dict(scene=self.config['scene'], robots=[self.config['robot']['robot_config']], env=self.config['og_sim']))
        self.og_env.scene.update_initial_state()
        for _ in range(10): og.sim.step()
        # robot vars #返回机器人变量
        self.robot = self.og_env.robots[0]
        # # 获取目标关节 ID
        # wrist_roll_joint = self.robot.arm_joint_names
        # print(f'wrist_roll_joint index is {wrist_roll_joint}')
        # #获得battery的orientation
        # for obj in self.og_env.scene.objects:
        #     if obj.name == 'battery_01':
        #         self.battery_orientation = obj.get_orientation()
        #         print(f'orientation of battery is :{self.battery_orientation}')
        # # print(f'robot:{self.robot}')
        # print(f'robot links:{self.robot.links}')
        # print(f'robot links wirst rool:{self.robot.links["wrist_roll_link"]}')
        # print(f'robot links wrist_roll_link 1:{self.robot.links["wrist_roll_link"].get_position_orientation()}')
        # # #设置旋转90度
        # self.robot.links["wrist_roll_link"].set_position_orientation(orientation=self.battery_orientation )
        # print(f'robot links wrist_roll_link 2:{self.robot.links["wrist_roll_link"].get_position_orientation()}')
        # print(f'link_pose:{self.robot.get_link_pose}')

        #机器人自由度：两部分，底座+机械臂
        dof_idx = np.concatenate([self.robot.trunk_control_idx,
                                  self.robot.arm_control_idx[self.robot.default_arm]])
        #设置机器人的关节位置
       # dof_idx:[ 2  4  6  7  8  9 10 11]

        self.reset_joint_pos = self.robot.reset_joint_pos[dof_idx]
        # positions = torch.Tensor([1.57])
        # indices = torch.Tensor([11])
        # self.robot.set_joint_positions(positions = positions, indices=indices)


        self.world2robot_homo = T.pose_inv(T.pose2mat(self.robot.get_position_orientation()))
        '''这段代码的作用是：获取机器人在世界坐标系中的位姿（位置和姿态）。将机器人在世界坐标系中的位姿转换为一个齐次变换矩阵。计算该变换矩阵的逆矩阵。将结果保存为 self.world2robot_homo，用于将任何点或物体从世界坐标系转换到机器人坐标系。'''
        '''表示从世界坐标系到机器人坐标系的齐次变换矩阵'''
        # initialize cameras 初始化相机
        self._initialize_cameras(self.config['camera'])
        self.last_og_gripper_action = 1.0


    # ======================================
    # = exposed functions
    # ======================================
    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        """
        open3d-based SDF computation SDF 通常指的是 Signed Distance Function（带符号距离函数）
        1. recursively get all usd prim and get their vertices and faces 循环获取prim（基本组件）的顶点和面
        2. compute SDF using open3d
        """
        #该方法用于计算场景中物体的有符号距离函数（Signed Distance Function，SDF）体素表示。SDF 是一种描述三维空间中每个点到物体表面距离的函数，通常用于碰撞检测、
        # 路径规划和形状分析等领域
        start = time.time()
        #创建一个排除列表，包含需要排除的物体名称，如墙壁、地板、天花板、机器人等。
        exclude_names = ['wall', 'floor', 'ceiling']
        if exclude_robot:
            exclude_names += ['fetch', 'robot']
        if exclude_obj_in_hand:
            assert self.config['robot']['robot_config']['grasping_mode'] in ['assisted', 'sticky'], "Currently only supported for assisted or sticky grasping"
            in_hand_obj = self.robot._ag_obj_in_hand[self.robot.default_arm]
            if in_hand_obj is not None:
                exclude_names.append(in_hand_obj.name.lower())
        trimesh_objects = []
        #遍历场景中的所有物体，收集它们的碰撞网格，并将其转换为 trimesh 对象。
        for obj in self.og_env.scene.objects:
            if any([name in obj.name.lower() for name in exclude_names]):
                continue
            for link in obj.links.values():
                for mesh in link.collision_meshes.values():
                    mesh_type = mesh.prim.GetPrimTypeInfo().GetTypeName()
                    if mesh_type == 'Mesh':
                        trimesh_object = mesh_prim_mesh_to_trimesh_mesh(mesh.prim)
                    else:
                        trimesh_object = mesh_prim_shape_to_trimesh_mesh(mesh.prim)
                    world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh.prim_path)
                    trimesh_object.apply_transform(world_pose_w_scale)
                    trimesh_objects.append(trimesh_object)
        # chain trimesh objects 将所有收集到的 trimesh 对象合并成一个场景网格。
        scene_mesh = trimesh.util.concatenate(trimesh_objects)
        # Create a scene and add the triangle mesh 使用合并后的场景网格创建一个 Open3D 场景，用于计算 SDF。
        scene = o3d.t.geometry.RaycastingScene()
        vertex_positions = scene_mesh.vertices
        triangle_indices = scene_mesh.faces
        vertex_positions = o3d.core.Tensor(vertex_positions, dtype=o3d.core.Dtype.Float32)
        triangle_indices = o3d.core.Tensor(triangle_indices, dtype=o3d.core.Dtype.UInt32)
        _ = scene.add_triangles(vertex_positions, triangle_indices)  # we do not need the geometry ID for mesh
        # create a grid 创建一个体素网格，覆盖场景的边界。
        shape = np.ceil((self.bounds_max - self.bounds_min) / resolution).astype(int)
        steps = (self.bounds_max - self.bounds_min) / shape
        grid = np.mgrid[self.bounds_min[0]:self.bounds_max[0]:steps[0],
                        self.bounds_min[1]:self.bounds_max[1]:steps[1],
                        self.bounds_min[2]:self.bounds_max[2]:steps[2]]
        grid = grid.reshape(3, -1).T
        # compute SDF  使用 Open3D 场景计算体素网格中每个点的有符号距离。
        sdf_voxels = scene.compute_signed_distance(grid.astype(np.float32))
        # convert back to np array  将计算得到的数据转换为numpy数组
        sdf_voxels = sdf_voxels.cpu().numpy()
        # open3d has flipped sign from our convention
        sdf_voxels = -sdf_voxels
        #将 SDF 值重塑为体素网格的形状。
        sdf_voxels = sdf_voxels.reshape(shape)
        self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] SDF voxels computed in {time.time() - start:.4f} seconds{bcolors.ENDC}')
        return sdf_voxels

    def get_cam_obs(self):
        #该方法用于获取环境中所有相机的观测数据
        self.last_cam_obs = dict()
        #便利相机列表，，每个相机都有一个唯一的标识符 cam_id
        for cam_id in self.cams:
            #调用每个相机的 get_obs 方法，获取该相机的观测数据，并将其存储在 last_cam_obs 字典中，键为相机的标识符 cam_id
            self.last_cam_obs[cam_id] = self.cams[cam_id].get_obs()  # each containing rgb, depth, points, seg
        return self.last_cam_obs #返回观测值
    
    def register_keypoints(self, keypoints):
        #该方法用于注册一组关键点（keypoints），这些关键点通常用于表示物体的关键位置或特征点。这些关键点在世界坐标系中的位置是已知的，
        # 并且该方法将这些关键点与场景中的物体进行关联。
        """
        Args:
            keypoints (np.ndarray): keypoints in the world frame of shape (N, 3) N是关键点的数量, 3表示(x, y, z)
        Returns:
            None
        Given a set of keypoints in the world frame, this function registers them so that their newest positions can be accessed later.
        """
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        self.keypoints = keypoints #存储输入的关键点。
        self._keypoint_registry = dict()  #将关键点索引映射到其最接近的 prim 路径和世界姿态
        self._keypoint2object = dict() #将关键点索引映射到其最接近的物体。
        #这个列表包含了在注册过程中应该被排除的物体名称。这些物体通常是静态的，不应该与关键点相关联。
        exclude_names = ['wall', 'floor', 'ceiling', 'table', 'fetch', 'robot']
        #便利关键点
        for idx, keypoint in enumerate(keypoints):
            closest_distance = np.inf
            #便利场景中的所有objects
            for obj in self.og_env.scene.objects:
                #如果在排除名单中，将其排除
                if any([name in obj.name.lower() for name in exclude_names]):
                    continue
                #遍历当前物体的link值
                for link in obj.links.values():
                    #遍历当前link的所有mesh
                    for mesh in link.visual_meshes.values():
                        #获取当前网格的原始路径 mesh_prim_path 和类型 mesh_type
                        mesh_prim_path = mesh.prim_path
                        mesh_type = mesh.prim.GetPrimTypeInfo().GetTypeName()
                        #如果mesh_type是Mesh，ze将其转换为trimesh
                        if mesh_type == 'Mesh':
                            trimesh_object = mesh_prim_mesh_to_trimesh_mesh(mesh.prim)
                        else: #如果是其他类型，则调用另一个函数将mesh转换为trimesh
                            trimesh_object = mesh_prim_shape_to_trimesh_mesh(mesh.prim)
                            #获取当前网格的世界姿态和比例 world_pose_w_scale，并应用到 trimesh 对象上。
                        world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh.prim_path)
                        #进行转换
                        trimesh_object.apply_transform(world_pose_w_scale)
                        #对 trimesh 对象进行采样，得到 1000 个采样点 points_transformed
                        points_transformed = trimesh_object.sample(1000)
                        
                        # find closest point 计算转换之后的点与key_points的二范数
                        dists = np.linalg.norm(points_transformed - keypoint, axis=1)
                        #找到距离最小的点
                        point = points_transformed[np.argmin(dists)]
                        distance = np.linalg.norm(point - keypoint)
                        #如果它小于之前找到的最接近点的距离 closest_distance，则更新 closest_distance、closest_prim_path、closest_point 和 closest_obj
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_prim_path = mesh_prim_path
                            closest_point = point
                            closest_obj = obj
        #closest_distance 存储了最接近点的距离，closest_prim_path 存储了最接近点所在的网格的原始路径，closest_point 存储了最接近点的坐标，
        # closest_obj 存储了最接近点所在的物体。
            self._keypoint_registry[idx] = (closest_prim_path, PoseAPI.get_world_pose(closest_prim_path))
            self._keypoint2object[idx] = closest_obj
            # overwrite the keypoint with the closest point
            self.keypoints[idx] = closest_point

    def get_keypoint_positions(self):
        #该方法用于获取已注册关键点在世界坐标系中的当前位置。
        """
        Args:
            None
        Returns:
            np.ndarray: keypoints in the world frame of shape (N, 3) #
        Given the registered keypoints, this function returns their current positions in the world frame.
        """
        assert hasattr(self, '_keypoint_registry') and self._keypoint_registry is not None, "Keypoints have not been registered yet."
        keypoint_positions = []
        #遍历 _keypoint_registry 字典中的每个条目，每个条目包含一个关键点的索引 idx、原始路径 prim_path 和初始姿态 init_pose。
        for idx, (prim_path, init_pose) in self._keypoint_registry.items():
            #将初始姿态转换为 4x4 的齐次变换矩阵。
            init_pose = T.pose2mat(init_pose)
            #计算中心转换位姿
            centering_transform = T.pose_inv(init_pose)
            #将关键点转换到中心点坐标系
            keypoint_centered = np.dot(centering_transform, np.append(self.keypoints[idx], 1))[:3]
            #获取当前网格的世界姿态，并转换为 4x4 的齐次变换矩阵。
            curr_pose = T.pose2mat(PoseAPI.get_world_pose(prim_path))
            #将关键点转换到中心点：
            keypoint = np.dot(curr_pose, np.append(keypoint_centered, 1))[:3]
            keypoint_positions.append(keypoint)
        # 返回关键点位置：
        return np.array(keypoint_positions)

    def get_object_by_keypoint(self, keypoint_idx):
        #通过关键点获取object
        """
        Args:
            keypoint_idx (int): the index of the keypoint
        Returns:
            pointer: the object that the keypoint is associated with
        Given the keypoint index, this function returns the name of the object that the keypoint is associated with.
        """
        assert hasattr(self, '_keypoint2object') and self._keypoint2object is not None, "Keypoints have not been registered yet."
        return self._keypoint2object[keypoint_idx]

    def get_collision_points(self, noise=True):
        #该方法用于获取机器人抓取器（gripper）和手中任何物体的碰撞点。这些碰撞点通常用于碰撞检测和避免碰撞的规划。
        """
        Get the points of the gripper and any object in hand.
        """
        # add gripper collision points
        collision_points = []
        for obj in self.og_env.scene.objects:
            if 'fetch' in obj.name.lower():
                for name, link in obj.links.items():
                    if 'gripper' in name.lower() or 'wrist' in name.lower():  # wrist_roll and wrist_flex
                        for collision_mesh in link.collision_meshes.values():
                            mesh_prim_path = collision_mesh.prim_path
                            mesh_type = collision_mesh.prim.GetPrimTypeInfo().GetTypeName()
                            if mesh_type == 'Mesh':
                                trimesh_object = mesh_prim_mesh_to_trimesh_mesh(collision_mesh.prim)
                            else:
                                trimesh_object = mesh_prim_shape_to_trimesh_mesh(collision_mesh.prim)
                            #获取世界坐标位姿
                            world_pose_w_scale = PoseAPI.get_world_pose_with_scale(mesh_prim_path)
                            #将获取到的object的mesh转换到世界坐标系下
                            trimesh_object.apply_transform(world_pose_w_scale)
                            #随机采样1000个点
                            points_transformed = trimesh_object.sample(1000)
                            # 将转换之后的点添加到列表中
                            collision_points.append(points_transformed)
        # add object in hand collision points 添加夹爪中的物体点
        in_hand_obj = self.robot._ag_obj_in_hand[self.robot.default_arm] #获取夹爪中的物体
        if in_hand_obj is not None:
            for link in in_hand_obj.links.values():
                for collision_mesh in link.collision_meshes.values():
                    mesh_type = collision_mesh.prim.GetPrimTypeInfo().GetTypeName()
                    if mesh_type == 'Mesh':
                        trimesh_object = mesh_prim_mesh_to_trimesh_mesh(collision_mesh.prim)
                    else:
                        trimesh_object = mesh_prim_shape_to_trimesh_mesh(collision_mesh.prim)
                    world_pose_w_scale = PoseAPI.get_world_pose_with_scale(collision_mesh.prim_path)
                    trimesh_object.apply_transform(world_pose_w_scale)
                    points_transformed = trimesh_object.sample(1000)
                    # add to collision points
                    collision_points.append(points_transformed)
        #将夹爪中的物体也添加到列表中，然后在第零个维度跟夹爪的关键点一起concat
        collision_points = np.concatenate(collision_points, axis=0)
        return collision_points

    def reset(self):
        #环境重置
        self.og_env.reset()
        #机器人重置
        self.robot.reset()
        positions = torch.Tensor([0.5])
        indices = torch.Tensor([11])
        self.robot.set_joint_positions(positions = positions, indices=indices)
        for _ in range(5): self._step()
        #打开机器人夹爪
        self.open_gripper()
        # moving arm to the side to unblock view 
        #获取机器人末端执行器的当前姿态（ee_pose），然后将其在 x 和 y 方向上移动一定的距离（-0.2 和 -0.1 米），
        # 并保持 z 方向不变。然后，它将新的姿态与一个无效的夹持器动作（get_gripper_null_action()）组合成一个动作数组，
        # 并使用 execute_action 方法来执行这个动作，精确地将末端执行器移动到新的位置。
        ee_pose = self.get_ee_pose()
        ee_pose[:3] += np.array([0.0, -0.2, -0.1])
        action = np.concatenate([ee_pose, [self.get_gripper_null_action()]])
        self.execute_action(action, precise=True)
        self.video_cache = []
        print(f'{bcolors.HEADER}Reset done.{bcolors.ENDC}')

    def is_grasping(self, candidate_obj=None):
        #判断机器人是否在夹取物体
        return self.robot.is_grasping(candidate_obj=candidate_obj) == IsGraspingState.TRUE

    def get_ee_pose(self):
        #获取机器人末端执行器的当前位姿
        ee_pos, ee_xyzw = (self.robot.get_eef_position(), self.robot.get_eef_orientation())
        #这行代码将位置向量 ee_pos 和方向向量 ee_xyzw 合并成一个 7 维向量
        ee_pose = np.concatenate([ee_pos, ee_xyzw])  # [7]
        return ee_pose

    def get_ee_pos(self):
        #返回末端执行器位置向量
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        #返回末端执行器旋转向量
        return self.get_ee_pose()[3:]
    
    def get_arm_joint_postions(self):
        assert isinstance(self.robot, Fetch), "The IK solver assumes the robot is a Fetch robot"
        #返回默认机械臂
        arm = self.robot.default_arm
        #self.robot.trunk_control_idx 和 self.robot.arm_control_idx[arm] 分别返回躯干和手臂的控制索引
        dof_idx = np.concatenate([self.robot.trunk_control_idx, self.robot.arm_control_idx[arm]])
        #这行代码调用 self.robot.get_joint_positions() 方法获取所有关节的位置，然后使用 dof_idx 索引数组来提取手臂关节的位置。
        arm_joint_pos = self.robot.get_joint_positions()[dof_idx]
        return arm_joint_pos

    def close_gripper(self):
        #关闭夹爪
        """
        Exposed interface: 1.0 for closed, -1.0 for open, 0.0 for no change
        Internal OG interface: 1.0 for open, 0.0 for closed
        """
        if self.last_og_gripper_action == 0.0:
            return
        action = np.zeros(12)
        action[10:] = [0, 0]  # gripper: float. 0. for closed, 1. for open.
        for _ in range(30):
            self._step(action)
        self.last_og_gripper_action = 0.0

    def open_gripper(self):
        #打开夹爪
        if self.last_og_gripper_action == 1.0:
            return
        action = np.zeros(12)
        action[10:] = [1, 1]  # gripper: float. 0. for closed, 1. for open.
        for _ in range(30):
            self._step(action)
        self.last_og_gripper_action = 1.0

    def get_last_og_gripper_action(self):
        return self.last_og_gripper_action
    
    def get_gripper_open_action(self):
        return -1.0
    
    def get_gripper_close_action(self):
        return 1.0
    
    def get_gripper_null_action(self):
        return 0.0
    
    def compute_target_delta_ee(self, target_pose):
        #用于计算机器人末端执行器（end-effector，EE）的目标位姿与当前位姿之间的差异
        target_pos, target_xyzw = target_pose[:3], target_pose[3:]
        ee_pose = self.get_ee_pose()
        ee_pos, ee_xyzw = ee_pose[:3], ee_pose[3:]
        pos_diff = np.linalg.norm(ee_pos - target_pos)
        rot_diff = angle_between_quats(ee_xyzw, target_xyzw)
        return pos_diff, rot_diff

    def execute_action(
            self,
            action,
            precise=True,
        ):
            #这个方法用于执行机器人的动作，包括移动到目标位姿和执行夹爪动作
            """
            Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

            Args:
            绝对目标位姿
                action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
                使用更加精确的移动
                precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
            Returns:
                tuple: A tuple containing the position and rotation errors after reaching the target pose.
            """
            if precise:
                pos_threshold = 0.03
                rot_threshold = 3.0
            else:
                pos_threshold = 0.10
                rot_threshold = 5.0
            action = np.array(action).copy()
            assert action.shape == (8,) #action是包含(x, y, z, qx, qy, qz, qw, gripper_action)的八元组
            target_pose = action[:7] #取前七个作为目标点的位姿
            gripper_action = action[7] #第八个是夹爪动作

            # ======================================
            # = status and safety check
            # ======================================
            #这部分代码检查目标位姿的前三个元素（位置）是否超出了工作空间的边界，如果超出，则将其限制在边界内。
            if np.any(target_pose[:3] < self.bounds_min) \
                 or np.any(target_pose[:3] > self.bounds_max):
                print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
                target_pose[:3] = np.clip(target_pose[:3], self.bounds_min, self.bounds_max)

            # ======================================
            # = interpolation
            # ======================================
            #这部分代码计算当前位姿和目标位姿之间的位置和旋转差异，并根据差异是否小于插值步长来决定是否需要插值。如果需要插值，则计算插值步骤并生成插值位姿序列
            current_pose = self.get_ee_pose()
            pos_diff = np.linalg.norm(current_pose[:3] - target_pose[:3])
            rot_diff = angle_between_quats(current_pose[3:7], target_pose[3:7])
            pos_is_close = pos_diff < self.interpolate_pos_step_size
            rot_is_close = rot_diff < self.interpolate_rot_step_size
            if pos_is_close and rot_is_close:
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Skipping interpolation{bcolors.ENDC}')
                pose_seq = np.array([target_pose])
            else:
                num_steps = get_linear_interpolation_steps(current_pose, target_pose, self.interpolate_pos_step_size, self.interpolate_rot_step_size)
                pose_seq = linear_interpolate_poses(current_pose, target_pose, num_steps)
                self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] Interpolating for {num_steps} steps{bcolors.ENDC}')

            # ======================================
            # = move to target pose
            # ======================================
            # move faster for intermediate poses
            #这部分代码将机器人移动到插值位姿序列中的每个位姿，最后移动到目标位姿，并计算位置和旋转误差
            intermediate_pos_threshold = 0.10
            intermediate_rot_threshold = 5.0
            for pose in pose_seq[:-1]:
                self._move_to_waypoint(pose, intermediate_pos_threshold, intermediate_rot_threshold)
            # move to the final pose with required precision
            pose = pose_seq[-1]
            self._move_to_waypoint(pose, pos_threshold, rot_threshold, max_steps=20 if not precise else 40) 
            # compute error
            pos_error, rot_error = self.compute_target_delta_ee(target_pose)
            self.verbose and print(f'\n{bcolors.BOLD}[environment.py | {get_clock_time()}] Move to pose completed (pos_error: {pos_error}, rot_error: {np.rad2deg(rot_error)}){bcolors.ENDC}\n')

            # ======================================
            # = apply gripper action
            # ======================================
            #这部分代码根据夹爪动作执行夹爪动作，包括打开夹爪、关闭夹爪和不执行动作。如果夹爪动作无效，则引发 ValueError 异常
            if gripper_action == self.get_gripper_open_action():
                self.open_gripper()
            elif gripper_action == self.get_gripper_close_action():
                self.close_gripper()
            elif gripper_action == self.get_gripper_null_action():
                pass
            else:
                raise ValueError(f"Invalid gripper action: {gripper_action}")
            
            return pos_error, rot_error
    
    def sleep(self, seconds):
        start = time.time()
        while time.time() - start < seconds:
            self._step()
    
    def save_video(self, save_path=None):
        save_dir = os.path.join(os.path.dirname(__file__), 'videos')
        os.makedirs(save_dir, exist_ok=True)
        if save_path is None:
            save_path = os.path.join(save_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.mp4')
        video_writer = imageio.get_writer(save_path, fps=30)
        for rgb in self.video_cache:
            video_writer.append_data(np.array(rgb))
        video_writer.close()
        return save_path

    # ======================================
    # = internal functions
    # ======================================
    def _check_reached_ee(self, target_pos, target_xyzw, pos_threshold, rot_threshold):
        """
        this is supposed to be for true ee pose (franka hand) in robot frame
        """
        #这个方法用于检查机器人的末端执行器（end-effector，EE）是否已经到达目标位姿。它通过比较当前位姿和目标位姿之间的位置和旋转差异来判断是否已经到达目标
        current_pos = self.robot.get_eef_position()
        current_xyzw = self.robot.get_eef_orientation()
        #将当前和目标方向的四元数转换为旋转矩阵，以便计算旋转差异。
        current_rotmat = T.quat2mat(current_xyzw)
        target_rotmat = T.quat2mat(target_xyzw)
        # calculate position delta
        #计算位置差异
        pos_diff = (target_pos - current_pos.cpu().numpy()).flatten()
        pos_error = np.linalg.norm(pos_diff)
        # calculate rotation delta
        rot_error = angle_between_rotmat(current_rotmat, target_rotmat)
        # print status
        self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Curr pose: {current_pos}, {current_xyzw} (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
        self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}]  Goal pose: {target_pos}, {target_xyzw} (pos_thres: {pos_threshold}, rot_thres: {rot_threshold}){bcolors.ENDC}')
        if pos_error < pos_threshold and rot_error < np.deg2rad(rot_threshold):
            self.verbose and print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose reached (pos_error: {pos_error.round(4)}, rot_error: {np.rad2deg(rot_error).round(4)}){bcolors.ENDC}')
            return True, pos_error, rot_error
        return False, pos_error, rot_error

    def _move_to_waypoint(self, target_pose_world, pos_threshold=0.02, rot_threshold=3.0, max_steps=10):
        #将机器人的末端执行器（end-effector，EE）移动到目标位姿。它通过不断检查当前位姿和目标位姿之间的差异，并根据差异调整机器人的动作，
        # 直到达到目标位姿或达到最大步数。
        pos_errors = []
        rot_errors = []
        count = 0
        while count < max_steps:
            #不断检查是否已经到达目标位姿。如果没有到达，它会将当前的位置误差和旋转误差添加到列表中，并继续执行循环。
            reached, pos_error, rot_error = self._check_reached_ee(target_pose_world[:3], target_pose_world[3:7], pos_threshold, rot_threshold)
            pos_errors.append(pos_error)
            rot_errors.append(rot_error)
            if reached:
                break
            # convert world pose to robot pose
            #将目标位姿从世界坐标系转换到机器人坐标系。
            target_pose_robot = np.dot(self.world2robot_homo, T.convert_pose_quat2mat(target_pose_world))
            # convert to relative pose to be used with the underlying controller
            #计算目标位姿和当前位姿之间的相对位置和相对旋转。
            relative_position = target_pose_robot[:3, 3] - self.robot.get_relative_eef_position().cpu().numpy()
            relative_quat = T.quat_distance(T.mat2quat(target_pose_robot[:3, :3]), self.robot.get_relative_eef_orientation())
            assert isinstance(self.robot, Fetch), "this action space is only for fetch"
            # 构建了一个动作向量，其中包含了相对位置、相对旋转和夹爪动作
            action = np.zeros(12)  # first 3 are base, which we don't use
            action[4:7] = relative_position
            action[7:10] = T.quat2axisangle(relative_quat)
            action[10:] = [self.last_og_gripper_action, self.last_og_gripper_action]
            # step the action
            _ = self._step(action=action)
            count += 1
        if count == max_steps:
            print(f'{bcolors.WARNING}[environment.py | {get_clock_time()}] OSC pose not reached after {max_steps} steps (pos_error: {pos_errors[-1].round(4)}, rot_error: {np.rad2deg(rot_errors[-1]).round(4)}){bcolors.ENDC}')

    def _step(self, action=None):
        #用于执行环境的一步操作，包括应用动作、更新环境状态、获取相机观测数据以及更新视频缓存
        if hasattr(self, 'disturbance_seq') and self.disturbance_seq is not None:
            next(self.disturbance_seq)
            #根据是否提供了动作参数来决定如何执行环境的一步操作。如果提供了动作，它会调用 self.og_env.step(action) 来执行动作；
            # 如果没有提供动作，它会调用 og.sim.step() 来执行一个默认的步骤
        if action is not None:
            self.og_env.step(action)
        else:
            og.sim.step()
        #获取环境中所有相机的观测数据，并从返回的观测数据中提取 RGB 图像。
        cam_obs = self.get_cam_obs()
        rgb = cam_obs[1]['rgb']
        if len(self.video_cache) < self.config['video_cache_size']:
            self.video_cache.append(rgb)
        else:
            self.video_cache.pop(0)
            self.video_cache.append(rgb)
        self.step_counter += 1

    def _initialize_cameras(self, cam_config):
        #这个方法用于初始化环境中的相机
        """
        ::param poses: list of tuples of (position, orientation) of the cameras
        """
        self.cams = dict()
        for cam_id in cam_config:
            cam_id = int(cam_id)
            self.cams[cam_id] = OGCamera(self.og_env, cam_config[cam_id])
        for _ in range(10): og.sim.render()