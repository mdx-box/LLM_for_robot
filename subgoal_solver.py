import numpy as np
import time
import copy
from scipy.optimize import dual_annealing, minimize
from scipy.interpolate import RegularGridInterpolator
import transform_utils as T
import torch
from utils import (
    transform_keypoints,
    calculate_collision_cost,
    normalize_vars,
    unnormalize_vars,
    farthest_point_sampling,
    consistency,
)
def objective(opt_vars, #优化变量
            og_bounds, #边界
            keypoints_centered, #中心化关键点
            keypoint_movable_mask, #关键点可移动mask
            goal_constraints, #目标约束
            path_constraints, #路径约束
            sdf_func, #sdf函数
            collision_points_centered, #中心碰撞点
            init_pose_homo, #初始姿态齐次矩阵
            ik_solver, #逆运动学求解器
            initial_joint_pos, #初始化关节位姿
            reset_joint_pos, #重置关节位姿
            is_grasp_stage, #是否处于抓取状态
            return_debug_dict=False):
#用于计算给定优化变量的目标函数值。这个目标函数值是通过考虑多个成本因素来确定的，包括碰撞成本、初始姿态成本、可达性成本、
# 重置正则化成本、抓取成本（如果处于抓取阶段）、目标约束违反成本和路径约束违反成本
    debug_dict = {}
    # unnormalize variables and do conversion
    #优化变量反归一化，并转换为姿态的齐次矩阵表示 （归一化处理的数据恢复到其原始的尺度或范围）
    opt_pose = unnormalize_vars(opt_vars, og_bounds)
    opt_pose_homo = T.pose2mat([opt_pose[:3], T.euler2quat(opt_pose[3:])])

    cost = 0
    # collision cost 中心点的碰撞点，则计算碰撞成本，并将其添加到总成本中。
    if collision_points_centered is not None:
        collision_cost = 0.8 * calculate_collision_cost(opt_pose_homo[None], sdf_func, collision_points_centered, 0.10)
        debug_dict['collision_cost'] = collision_cost
        cost += collision_cost

    # stay close to initial pose 计算初始姿态成本，并将其添加到总成本中。
    init_pose_cost = 1.0 * consistency(opt_pose_homo[None], init_pose_homo[None], rot_weight=1.5)
    debug_dict['init_pose_cost'] = init_pose_cost
    cost += init_pose_cost

    # reachability cost (approximated by number of IK iterations + regularization from reset joint pos) 计算 IK 求解的可达性成本，并将其添加到总成本中。
    max_iterations = 20
    ik_result = ik_solver.solve(
                    opt_pose_homo,
                    max_iterations=max_iterations,
                    initial_joint_pos=initial_joint_pos,
                )
    ik_cost = 20.0 * (ik_result.num_descents / max_iterations)
    debug_dict['ik_feasible'] = ik_result.success
    debug_dict['ik_pos_error'] = ik_result.position_error
    debug_dict['ik_cost'] = ik_cost
    cost += ik_cost
    # # 确保在调用之前检查并转换类型
    # if isinstance(ik_result.cspace_position[:-1], torch.Tensor):
    #     ik_result_cspace_position_np = ik_result.cspace_position.cpu().numpy()
    # else:
    #     ik_result_cspace_position_np = ik_result.cspace_position

    # if isinstance(reset_joint_pos, torch.Tensor):
    #     reset_joint_pos_np = reset_joint_pos.cpu().numpy()
    # else:
    #     reset_joint_pos_np = reset_joint_pos    
    #如果 IK 求解成功，则计算重置正则化成本，并将其添加到总成本中。
    if ik_result.success:
        reset_reg = np.linalg.norm(ik_result.cspace_position[:-1] - reset_joint_pos[:-1].cpu().numpy())
        reset_reg = np.clip(reset_reg, 0.0, 3.0)
    else:
        reset_reg = 3.0
    reset_reg_cost = 0.2 * reset_reg
    debug_dict['reset_reg_cost'] = reset_reg_cost
    cost += reset_reg_cost

    # grasp metric (better performance if using anygrasp or force-based grasp metrics)
    #如果处于抓取阶段，则计算抓取成本，并将其添加到总成本中。
    if is_grasp_stage:
        preferred_dir = np.array([0, 0, -1]) 
        grasp_cost = -np.dot(opt_pose_homo[:3, 0], preferred_dir) + 1  # [0, 1]
        grasp_cost = 10.0 * grasp_cost
        debug_dict['grasp_cost'] = grasp_cost
        cost += grasp_cost

    # goal constraint violation cost 如果存在目标约束，则计算目标约束违反成本，并将其添加到总成本中。
    debug_dict['subgoal_constraint_cost'] = None
    debug_dict['subgoal_violation'] = None
    if goal_constraints is not None and len(goal_constraints) > 0:
        subgoal_constraint_cost = 0
        transformed_keypoints = transform_keypoints(opt_pose_homo, keypoints_centered, keypoint_movable_mask)
        subgoal_violation = []
        for constraint in goal_constraints:
            violation = constraint(transformed_keypoints[0], transformed_keypoints[1:])
            subgoal_violation.append(violation)
            subgoal_constraint_cost += np.clip(violation, 0, np.inf)
        subgoal_constraint_cost = 200.0*subgoal_constraint_cost
        debug_dict['subgoal_constraint_cost'] = subgoal_constraint_cost
        debug_dict['subgoal_violation'] = subgoal_violation
        cost += subgoal_constraint_cost
    
    # path constraint violation cost 如果存在路径约束，则计算路径约束违反成本，并将其添加到总成本中
    debug_dict['path_violation'] = None
    if path_constraints is not None and len(path_constraints) > 0:
        path_constraint_cost = 0
        transformed_keypoints = transform_keypoints(opt_pose_homo, keypoints_centered, keypoint_movable_mask)
        path_violation = []
        for constraint in path_constraints:
            violation = constraint(transformed_keypoints[0], transformed_keypoints[1:])
            path_violation.append(violation)
            path_constraint_cost += np.clip(violation, 0, np.inf)
        path_constraint_cost = 200.0*path_constraint_cost
        debug_dict['path_constraint_cost'] = path_constraint_cost
        debug_dict['path_violation'] = path_violation
        cost += path_constraint_cost

    debug_dict['total_cost'] = cost

    if return_debug_dict:
        return cost, debug_dict

    return cost  #返回全部cost


class SubgoalSolver:
    def __init__(self, config, ik_solver, reset_joint_pos):
        self.config = config #配置文件
        self.ik_solver = ik_solver #逆运动学求解器
        self.reset_joint_pos = reset_joint_pos #重置关节位姿
        self.last_opt_result = None #最后一次优化结果
        # warmup
        self._warmup()

    def _warmup(self):
        #末端执行器的位姿[x, y, z, qx, qy, qz, qw] 在开始正式的优化求解之前，先进行一次模拟求解，以确保所有的组件和设置都正常工作
        ee_pose = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1]) 
        #关键点的位置
        keypoints = np.random.rand(10, 3)
        #探测关键点是否在机器人抓取的物体上
        keypoint_movable_mask = np.random.rand(10) > 0.5
        #子目标约束函数
        goal_constraints = []
        #路径约束函数
        path_constraints = []
        #环境的距离[X, Y, Z]
        sdf_voxels = np.zeros((10, 10, 10))
        #物体点云[N, 3]
        collision_points = np.random.rand(100, 3)
        
        self.solve(ee_pose, keypoints, keypoint_movable_mask, goal_constraints, path_constraints, sdf_voxels, collision_points, True, None, from_scratch=True)
        self.last_opt_result = None

    def _setup_sdf(self, sdf_voxels):
        # create callable sdf function with interpolation 
        #设置一个可调用的有符号距离场（Signed Distance Field，SDF）函数，该函数通过插值来近似表示环境的SDF
        #分别表示SDF体素在三个维度上的坐标范围。这些数组将用于构建插值函数的网格
        x = np.linspace(self.config['bounds_min'][0], self.config['bounds_max'][0], sdf_voxels.shape[0])
        y = np.linspace(self.config['bounds_min'][1], self.config['bounds_max'][1], sdf_voxels.shape[1])
        z = np.linspace(self.config['bounds_min'][2], self.config['bounds_max'][2], sdf_voxels.shape[2])
        #创建插值函数
        sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
        return sdf_func

    def _check_opt_result(self, opt_result, debug_dict):
        # accept the opt_result if it's only terminated due to iteration limit
        #检查优化结果是否有效，包括检查目标约束和路径约束是否满足，以及逆运动学（IK）是否可行 
        #检查优化结果是否因为达到最大迭代次数而终止
        if (not opt_result.success and ('maximum' in opt_result.message.lower() or 'iteration' in opt_result.message.lower() or 'not necessarily' in opt_result.message.lower())):
            opt_result.success = True
        elif not opt_result.success:
            opt_result.message += '; invalid solution'
        # check whether goal constraints are satisfied 检查目标约束是否满足
        if debug_dict['subgoal_violation'] is not None:
            goal_constraints_results = np.array(debug_dict['subgoal_violation'])
            opt_result.message += f'; goal_constraints_results: {goal_constraints_results} (higher is worse)' 
            #all() 用于判断可迭代对象中的所有元素是否都为真值
            goal_constraints_satisfied = all([violation <= self.config['constraint_tolerance'] for violation in goal_constraints_results])
            if not goal_constraints_satisfied:
                opt_result.success = False
                opt_result.message += f'; goal not satisfied'
        # check whether path constraints are satisfied 检查路径约束是否满足
        if debug_dict['path_violation'] is not None:
            path_constraints_results = np.array(debug_dict['path_violation'])
            opt_result.message += f'; path_constraints_results: {path_constraints_results}'
            path_constraints_satisfied = all([violation <= self.config['constraint_tolerance'] for violation in path_constraints_results])
            if not path_constraints_satisfied:
                opt_result.success = False
                opt_result.message += f'; path not satisfied'
        # check whether ik is feasible 检查逆运动学是否可行
        if 'ik_feasible' in debug_dict and not debug_dict['ik_feasible']:
            opt_result.success = False
            opt_result.message += f'; ik not feasible'
        return opt_result
    
    def _center_collision_points_and_keypoints(self, ee_pose_homo, collision_points, keypoints, keypoint_movable_mask):
        #将碰撞点和关键点转换到以末端执行器位姿为中心的坐标系中
        centering_transform = np.linalg.inv(ee_pose_homo)
        #将碰撞点从世界坐标系转换到以末端执行器为中心的坐标系4
        #通过矩阵乘法将碰撞点的位置向量乘以中心化变换矩阵的前3行3列的转置，然后加上中心化变换矩阵的前3行第4列的平移向量。
        collision_points_centered = np.dot(collision_points, centering_transform[:3, :3].T) + centering_transform[:3, 3]
        #碰撞点转换到以末端执行器为中心的坐标系
        keypoints_centered = transform_keypoints(centering_transform, keypoints, keypoint_movable_mask)
        return collision_points_centered, keypoints_centered #返回转换后的碰撞点和关键点

    def solve(self,
            ee_pose, #末端执行器位姿
            keypoints, #关键点
            keypoint_movable_mask, #一个布尔数组，指示哪些关键点是可移动的
            goal_constraints, #目标约束函数的列表
            path_constraints,# 路径约束函数的列表
            sdf_voxels, #环境的有符号距离场
            collision_points, #物体点云 (N, 3)
            is_grasp_stage, #当前stage是否是抓取stage
            initial_joint_pos, #初始机器人关节位置
            from_scratch=False, #是否从头开始
            ):
        """
        Args:
            - ee_pose (np.ndarray): [7], [x, y, z, qx, qy, qz, qw] end effector pose.
            - keypoints (np.ndarray): [M, 3] keypoint positions.
            - keypoint_movable_mask (bool): [M] boolean array indicating whether the keypoint is on the grasped object.
            - goal_constraints (List[Callable]): subgoal constraint functions.
            - path_constraints (List[Callable]): path constraint functions.
            - sdf_voxels (np.ndarray): [X, Y, Z] signed distance field of the environment.
            - collision_points (np.ndarray): [N, 3] point cloud of the object.
            - is_grasp_stage (bool): whether the current stage is a grasp stage.
            - initial_joint_pos (np.ndarray): [N] initial joint positions of the robot.
            - from_scratch (bool): whether to start from scratch.
        Returns:
            - result (scipy.optimize.OptimizeResult): optimization result.
            - debug_dict (dict): debug information.
        """
        # downsample collision points
        #如果碰撞点云存在且点数超过配置中的最大碰撞点数，则使用最远点采样算法对碰撞点进行采样。
        if collision_points is not None and collision_points.shape[0] > self.config['max_collision_points']:
            collision_points = farthest_point_sampling(collision_points, self.config['max_collision_points'])
        sdf_func = self._setup_sdf(sdf_voxels) #设置一个可调用的有符号距离场函数。
        # ====================================
        # = setup bounds and initial guess
        # ====================================
        ee_pose = ee_pose.astype(np.float64)
        ee_pose_homo = T.pose2mat([ee_pose[:3], ee_pose[3:]])
        #将末端执行器位姿转换为欧拉角表示
        ee_pose_euler = np.concatenate([ee_pose[:3], T.quat2euler(ee_pose[3:])])
        # normalize opt variables to [0, 1] 标准化变量到0-1之间
        pos_bounds_min = self.config['bounds_min']
        pos_bounds_max = self.config['bounds_max']
        rot_bounds_min = np.array([-np.pi, -np.pi, -np.pi])  # euler angles
        rot_bounds_max = np.array([np.pi, np.pi, np.pi])  # euler angles
        #将位置和旋转边界合并为一个列表，用于归一化变量的范围。
        og_bounds = [(b_min, b_max) for b_min, b_max in zip(np.concatenate([pos_bounds_min, rot_bounds_min]), np.concatenate([pos_bounds_max, rot_bounds_max]))]
        bounds = [(-1, 1)] * len(og_bounds) #将每个边界转换为 [-1, 1] 的范围，这是优化算法中常用的归一化方法，以便于优化算法的处理
        if not from_scratch and self.last_opt_result is not None:
            init_sol = self.last_opt_result.x
        else:
            init_sol = normalize_vars(ee_pose_euler, og_bounds)  # start from the current pose
            from_scratch = True

        # ====================================
        # = other setup
        # ====================================
        # 计算碰撞点和关键点的中心点和变换后的关键点
        collision_points_centered, keypoints_centered = self._center_collision_points_and_keypoints(ee_pose_homo, collision_points, keypoints, keypoint_movable_mask)
        aux_args = (og_bounds,
                    keypoints_centered,
                    keypoint_movable_mask,
                    goal_constraints,
                    path_constraints,
                    sdf_func,
                    collision_points_centered,
                    ee_pose_homo,
                    self.ik_solver,
                    initial_joint_pos,
                    self.reset_joint_pos,
                    is_grasp_stage)

        # ====================================
        # = solve optimization
        # ====================================
        start = time.time()
        # use global optimization for the first iteration
        if from_scratch: #如果是从头开始计算
            #dual_annealing: 使用Dual Annealing发现函数的全局最小值

            opt_result = dual_annealing(
                func=objective,
                bounds=bounds,
                args=aux_args,
                maxfun=self.config['sampling_maxfun'],
                x0=init_sol,
                no_local_search=False,
                minimizer_kwargs={
                    'method': 'SLSQP',
                    'options': self.config['minimizer_options'],
                },
            )
        # use gradient-based local optimization for the following iterations
        else: #如果不是从头开始计算
            opt_result = minimize(
                fun=objective,
                x0=init_sol,
                args=aux_args,
                bounds=bounds,
                method='SLSQP',
                options=self.config['minimizer_options'],
            )
        solve_time = time.time() - start

        # ====================================
        # = post-process opt_result
        # ====================================
        if isinstance(opt_result.message, list):
            opt_result.message = opt_result.message[0]
        # rerun to get debug info
        _, debug_dict = objective(opt_result.x, *aux_args, return_debug_dict=True)
        debug_dict['sol'] = opt_result.x
        debug_dict['msg'] = opt_result.message
        debug_dict['solve_time'] = solve_time
        debug_dict['from_scratch'] = from_scratch
        debug_dict['type'] = 'subgoal_solver'
        # unnormailze
        sol = unnormalize_vars(opt_result.x, og_bounds)
        sol = np.concatenate([sol[:3], T.euler2quat(sol[3:])])
        opt_result = self._check_opt_result(opt_result, debug_dict)
        # cache opt_result for future use if successful
        if opt_result.success:
            self.last_opt_result = copy.deepcopy(opt_result)
        return sol, debug_dict