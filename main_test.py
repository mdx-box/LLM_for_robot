import torch
import numpy as np
import json
import os
import argparse
import time
from environment import ReKepOGEnv
from keypoint_proposal import KeypointProposer
from constraint_generation import ConstraintGenerator
from ik_solver import IKSolver
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
from visualizer import Visualizer
import transform_utils as T
from omnigibson.robots.fetch import Fetch
import cv2
from utils import (
    bcolors,
    get_config,
    load_functions_from_txt,
    get_linear_interpolation_steps,
    spline_interpolate_poses,
    get_callable_grasping_cost_fn,
    print_opt_debug_dict,
)

class Main:
    def __init__(self, scene_file, visualize=False):
        global_config = get_config(config_path="./configs/config.yaml")
        self.config = global_config['main']
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.visualize = visualize
        # set random seed
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        # initialize keypoint proposer and constraint generator
        self.keypoint_proposer = KeypointProposer(global_config['keypoint_proposer'])
        #根据图像生成查询message，然后将返回的查询结果的stage 1 2 3解析出来（是三个阶段的约束函数），将其放入到指定文件夹中
        self.constraint_generator = ConstraintGenerator(global_config['constraint_generator'])
        # initialize environment 初始化环境
        self.env = ReKepOGEnv(global_config['env'], scene_file, verbose=False)
        # setup ik solver (for reachability cost)
        assert isinstance(self.env.robot, Fetch), "The IK solver assumes the robot is a Fetch robot"
        #运动学正逆解
        ik_solver = IKSolver(
            robot_description_path=self.env.robot.robot_arm_descriptor_yamls[self.env.robot.default_arm], #fetch的描述地址
            robot_urdf_path=self.env.robot.urdf_path, #fetch的urdf描述文件
            eef_name=self.env.robot.eef_link_names[self.env.robot.default_arm],  #末端执行器name
            reset_joint_pos=self.env.reset_joint_pos, #joint默认位姿
            world2robot_homo=self.env.world2robot_homo, #世界坐标系到机器人坐标系的齐次变换矩阵
        )
        # initialize solvers 初始化子目标求解器
        self.subgoal_solver = SubgoalSolver(global_config['subgoal_solver'], ik_solver, self.env.reset_joint_pos)
        #初始化路径求解器
        self.path_solver = PathSolver(global_config['path_solver'], ik_solver, self.env.reset_joint_pos)
        # initialize visualizer
        if self.visualize:
            self.visualizer = Visualizer(global_config['visualizer'], self.env)

    def perform_task(self, instruction, rekep_program_dir=None, disturbance_seq=None):
        #重置环境、获取相机观测、生成关键点和约束、以及执行任务。
        #重置环境，将环境恢复到初始状态。
        self.env.reset()
        ##获取环境中的相机观测，包括 RGB 图像、点云数据和分割掩码
        cam_obs = self.env.get_cam_obs()
        rgb = cam_obs[self.config['vlm_camera']]['rgb']
        points = cam_obs[self.config['vlm_camera']]['points']
        mask = cam_obs[self.config['vlm_camera']]['seg']

        # 创建数据保存目录
        save_dir = os.path.join(os.path.dirname(__file__), 'camera_data')
        os.makedirs(save_dir, exist_ok=True)

        # 保存RGB图像为JPG
        rgb_path = os.path.join(save_dir, 'rgb_image.jpg')
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb.numpy(), cv2.COLOR_RGB2BGR))

        # 保存点云和掩码数据为NPY
        points_path = os.path.join(save_dir, 'points.npy')
        mask_path = os.path.join(save_dir, 'mask.npy')
        # 检查points和mask是否为PyTorch张量，如果是则转换为NumPy数组
        points_data = points.cpu().numpy() if torch.is_tensor(points) else points
        mask_data = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        np.save(points_path, points_data)
        np.save(mask_path, mask_data)

        print(f'Camera data saved to {save_dir}')
        # print(f'rgb is:{rgb}')
        points = cam_obs[self.config['vlm_camera']]['points']
        mask = cam_obs[self.config['vlm_camera']]['seg']
        # ====================================
        # = keypoint proposal and constraint generation
        # ====================================
        if rekep_program_dir is None:
            #获取关键点和投影图像
            keypoints, projected_img = self.keypoint_proposer.get_keypoints(rgb, points, mask)
            print(f'{bcolors.HEADER}Got {len(keypoints)} proposed keypoints{bcolors.ENDC}')
            if self.visualize:
                self.visualizer.show_img(projected_img)
                #将数据变为元数据
            metadata = {'init_keypoint_positions': keypoints, 'num_keypoints': len(keypoints)}
            #根据给定的图像和指令生成约束条件，并将这些约束条件保存到文件中
            rekep_program_dir = self.constraint_generator.generate(projected_img, instruction, metadata)
            print(f'{bcolors.HEADER}Constraints generated{bcolors.ENDC}')
        # ====================================
        # = execute
        # ====================================
        self._execute(rekep_program_dir, disturbance_seq)

    def _update_disturbance_seq(self, stage, disturbance_seq):
        #用于更新环境中的扰动序列，以便在执行任务时模拟干扰
        if disturbance_seq is not None:
            if stage in disturbance_seq and not self.applied_disturbance[stage]:
                # set the disturbance sequence, the generator will yield and instantiate one disturbance function for each env.step until it is exhausted
                self.env.disturbance_seq = disturbance_seq[stage](self.env)
                self.applied_disturbance[stage] = True

    def _execute(self, rekep_program_dir, disturbance_seq=None):
        #包括加载元数据、注册关键点、加载约束、更新扰动序列、获取优化计划和执行动作。
        # load metadata
        print(f'rekep_program_dir: {rekep_program_dir}')
        with open(os.path.join(rekep_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        self.applied_disturbance = {stage: False for stage in range(1, self.program_info['num_stages'] + 1)} #三个全部设置为false
        # register keypoints to be tracked
        self.env.register_keypoints(self.program_info['init_keypoint_positions'])
        # load constraints
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            for constraint_type in ['subgoal', 'path']:
                load_path = os.path.join(rekep_program_dir, f'stage{stage}_{constraint_type}_constraints.txt')
                get_grasping_cost_fn = get_callable_grasping_cost_fn(self.env)  # special grasping function for VLM to call
                stage_dict[constraint_type] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict
        
        # bookkeeping of which keypoints can be moved in the optimization
        #初始化一个布尔数组，用于标记哪些关键点可以在优化中移动
        self.keypoint_movable_mask = np.zeros(self.program_info['num_keypoints'] + 1, dtype=bool)
        self.keypoint_movable_mask[0] = True  # first keypoint is always the ee, so it's movable

        # main loop
        self.last_sim_step_counter = -np.inf
        self._update_stage(1) #将当前阶段更新为 1
        while True:
            #获取场景关键点
            scene_keypoints = self.env.get_keypoint_positions()
            #将场景关键点以及机器人末端执行器的位姿组合成一个数组
            self.keypoints = np.concatenate([[self.env.get_ee_pos()], scene_keypoints], axis=0)  # first keypoint is always the ee
            #当前末端执行器的位姿和关节位置
            self.curr_ee_pose = self.env.get_ee_pose()
            self.curr_joint_pos = self.env.get_arm_joint_postions()
            
            # 保存末端执行器位姿到文件
            # with open('end_effector/eef_position.txt', 'a') as f:
            #     # 添加时间戳和阶段信息
            #     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            #     # f.write(f"Time: {timestamp}, Stage: {self.stage}\n")
            #     # # 保存位置和姿态数据
            #     f.write(f"Position: {self.curr_ee_pose[:3]}\n")
            #     f.write(f"Orientation: {self.curr_ee_pose[3:]}\n\n")
            #     f.flush()  # 立即将数据写入文件
            #获取sdf体素
            self.sdf_voxels = self.env.get_sdf_voxels(self.config['sdf_voxel_size'])
            #获取碰撞点
            self.collision_points = self.env.get_collision_points()
            # ====================================
            # = decide whether to backtrack
            # ====================================
            backtrack = False
            #如果当前阶段大于 1，机器人会检查路径约束是否被违反。如果违反，它会回溯到之前的阶段，直到找到一个满足所有约束的阶段。
            if self.stage > 1:
                path_constraints = self.constraint_fns[self.stage]['path']
                for constraints in path_constraints:
                    violation = constraints(self.keypoints[0], self.keypoints[1:])
                    if violation > self.config['constraint_tolerance']:
                        backtrack = True
                        break
            if backtrack:
                # determine which stage to backtrack to based on constraints
                for new_stage in range(self.stage - 1, 0, -1):
                    path_constraints = self.constraint_fns[new_stage]['path']
                    # if no constraints, we can safely backtrack
                    if len(path_constraints) == 0:
                        break
                    # otherwise, check if all constraints are satisfied
                    all_constraints_satisfied = True
                    for constraints in path_constraints:
                        violation = constraints(self.keypoints[0], self.keypoints[1:])
                        #检查是否需要回溯，如果当前阶段的路径约束被违反，则回溯到之前的阶段。
                        if violation > self.config['constraint_tolerance']:
                            all_constraints_satisfied = False
                            break
                    if all_constraints_satisfied:   
                        break
                print(f"{bcolors.HEADER}[stage={self.stage}] backtrack to stage {new_stage}{bcolors.ENDC}")
                self._update_stage(new_stage)
            else:
                # apply disturbance
                self._update_disturbance_seq(self.stage, disturbance_seq)
                # ====================================
                # = get optimized plan
                # ====================================
                #如果模拟步骤计数器没有更新，机器人会打印一条警告信息。然后，它会获取下一个子目标和路径，并更新动作队列
                if self.last_sim_step_counter == self.env.step_counter:
                    print(f"{bcolors.WARNING}sim did not step forward within last iteration (HINT: adjust action_steps_per_iter to be larger or the pos_threshold to be smaller){bcolors.ENDC}")
                #如果不需要回溯，则获取下一个子目标和路径，并更新动作队列。
                next_subgoal = self._get_next_subgoal(from_scratch=self.first_iter)
                next_path = self._get_next_path(next_subgoal, from_scratch=self.first_iter)
                self.first_iter = False
                self.action_queue = next_path.tolist()
                self.last_sim_step_counter = self.env.step_counter

                # ====================================
                # = execute
                # ====================================
                # determine if we proceed to the next stage
                #在这个循环中，机器人不断地从动作队列中取出动作并执行，直到动作队列为空。如果当前阶段是抓取或释放阶段，它会执行相应的抓取或释放动作。
                # 如果所有阶段都完成了，它会保存视频并返回
                count = 0
                while len(self.action_queue) > 0 and count < self.config['action_steps_per_iter']:
                    next_action = self.action_queue.pop(0)
                    precise = len(self.action_queue) == 0
                    self.env.execute_action(next_action, precise=precise)
                    # 每次执行动作后记录末端执行器位姿
                    with open('end_effector/txt/eef_position_eef_position_chassis0.15._battery_-0.15.txt', 'a') as f:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        f.write(f"Time: {timestamp}, Stage: {self.stage}\n")
                        f.write(f"Position: {self.env.get_ee_pose()[:3]}\n")
                        f.write(f"Orientation: {self.env.get_ee_pose()[3:]}\n\n")
                        f.flush()
                    count += 1
                if len(self.action_queue) == 0:
                    if self.is_grasp_stage:
                        self._execute_grasp_action()
                    elif self.is_release_stage:
                        self._execute_release_action()
                    # if completed, save video and return
                    if self.stage == self.program_info['num_stages']: 
                        self.env.sleep(2.0)
                        save_path = self.env.save_video()
                        print(f"{bcolors.OKGREEN}Video saved to {save_path}\n\n{bcolors.ENDC}")
                        return
                    # progress to next stage
                    self._update_stage(self.stage + 1)

    def _get_next_subgoal(self, from_scratch):
        #用于计算机器人在当前阶段的下一个子目标姿态
        #获取当前阶段的子目标约束。
        subgoal_constraints = self.constraint_fns[self.stage]['subgoal']
        path_constraints = self.constraint_fns[self.stage]['path']
        #子目标求解器，获取优化后的子目标姿态和调试信息。
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.keypoint_movable_mask,
                                                            subgoal_constraints,
                                                            path_constraints,
                                                            self.sdf_voxels,
                                                            self.collision_points,
                                                            self.is_grasp_stage,
                                                            self.curr_joint_pos,
                                                            from_scratch=from_scratch)
        #子目标姿态从四元数表示转换为齐次矩阵表示。
        subgoal_pose_homo = T.convert_pose_quat2mat(subgoal_pose)
        # if grasp stage, back up a bit to leave room for grasping
        #如果当前阶段是抓取阶段，这行代码会将子目标位置向后调整一个抓取深度的一半，以留出抓取的空间
        if self.is_grasp_stage:
            subgoal_pose[:3] += subgoal_pose_homo[:3, :3] @ np.array([-self.config['grasp_depth'] / 2.0, 0, 0])
        #将当前阶段添加到调试字典中，并打印出调试信息。
        debug_dict['stage'] = self.stage
        print_opt_debug_dict(debug_dict)
        if self.visualize:
            self.visualizer.visualize_subgoal(subgoal_pose)
        return subgoal_pose #返回子目标姿态

    def _get_next_path(self, next_subgoal, from_scratch):
        #获取当前阶段的路径约束
        path_constraints = self.constraint_fns[self.stage]['path']
        #路径求解器，获取优化后的路径和调试信息。
        path, debug_dict = self.path_solver.solve(self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.keypoint_movable_mask,
                                                    path_constraints,
                                                    self.sdf_voxels,
                                                    self.collision_points,
                                                    self.curr_joint_pos,
                                                    from_scratch=from_scratch)
        print_opt_debug_dict(debug_dict)
        processed_path = self._process_path(path)
        if self.visualize:
            self.visualizer.visualize_path(processed_path)
        return processed_path

    def _process_path(self, path):
        # spline interpolate the path from the current ee pose
        #用于处理路径，包括对路径进行样条插值，并添加相应的夹爪动作。
        #首先将当前末端执行器的姿态（self.curr_ee_pose）与输入的路径点（path）连接起来，形成一个完整的控制点数组（full_control_points）
        full_control_points = np.concatenate([
            self.curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        #使用 get_linear_interpolation_steps 函数计算出在给定的位置和旋转插值步长下，从第一个控制点到最后一个控制点需要的插值步数（num_steps）。
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        #使用 spline_interpolate_poses 函数对这些控制点进行样条插值，生成一个密集的路径（dense_path）
        dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        #创建一个与密集路径长度相同的动作序列数组（ee_action_seq），并将其初始化为零。
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        #它将密集路径的姿态部分（前七个元素）复制到动作序列的相应位置
        ee_action_seq[:, :7] = dense_path
        #将环境的夹爪空动作（self.env.get_gripper_null_action()）添加到动作序列的最后一个元素，即夹爪动作
        ee_action_seq[:, 7] = self.env.get_gripper_null_action()
        return ee_action_seq

    def _update_stage(self, stage):
        # update stage 更新机器人的当前阶段，并根据新阶段的类型（抓取或释放）执行相应的操作
        self.stage = stage
        self.is_grasp_stage = self.program_info['grasp_keypoints'][self.stage - 1] != -1
        self.is_release_stage = self.program_info['release_keypoints'][self.stage - 1] != -1
        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage + self.is_release_stage <= 1, "Cannot be both grasp and release stage"
        if self.is_grasp_stage:  # ensure gripper is open for grasping stage
            self.env.open_gripper()
        # clear action queue
        self.action_queue = []
        # update keypoint movable mask
        self._update_keypoint_movable_mask()
        self.first_iter = True

    def _update_keypoint_movable_mask(self):
        #用于更新关键点的可移动掩码，该掩码用于指示哪些关键点可以在当前阶段移动
        for i in range(1, len(self.keypoint_movable_mask)):  # first keypoint is ee so always movable
            keypoint_object = self.env.get_object_by_keypoint(i - 1)
            #根据环境对象 self.env 的 is_grasping 方法的返回值来更新关键点的可移动掩码。如果物体被抓取，则对应的关键点不可移动，否则可移动。
            self.keypoint_movable_mask[i] = self.env.is_grasping(keypoint_object)

    def _execute_grasp_action(self):
        #用于执行抓取动作，包括计算预抓取姿态和抓取姿态，并将抓取动作发送到环境中执行
        #获取末端执行器位姿
        pregrasp_pose = self.env.get_ee_pose()
        grasp_pose = pregrasp_pose.copy()
        #计算抓取姿态。它首先将预抓取姿态的前三个元素（位置）加上一个向量，这个向量是通过将预抓取姿态的四元数旋转矩阵与一个表示抓取深度的向量相乘得到的。
        # 这样，抓取姿态就比预抓取姿态更靠近物体。
        grasp_pose[:3] += T.quat2mat(pregrasp_pose[3:]) @ np.array([self.config['grasp_depth'], 0, 0])
        #构建抓取动作。它将抓取姿态和夹爪关闭动作连接起来，形成一个包含姿态和夹爪动作的动作数组。
        grasp_action = np.concatenate([grasp_pose, [self.env.get_gripper_close_action()]])
        self.env.execute_action(grasp_action, precise=True)
    
    def _execute_release_action(self):
        self.env.open_gripper()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pen', help='task to perform')
    parser.add_argument('--use_cached_query', action='store_true', help='instead of querying the VLM, use the cached query')
    parser.add_argument('--apply_disturbance', action='store_true', help='apply disturbance to test the robustness')
    parser.add_argument('--visualize', action='store_true', help='visualize each solution before executing (NOTE: this is blocking and needs to press "ESC" to continue)')
    args = parser.parse_args()

    if args.apply_disturbance:
        assert args.task == 'pen' and args.use_cached_query, 'disturbance sequence is only defined for cached scenario'

    # ====================================
    # = pen task disturbance sequence
    # ====================================
    def stage1_disturbance_seq(env):
        """
        Move the pen in stage 0 when robot is trying to grasp the pen
        用于模拟在机器人试图抓取笔时，对笔进行干扰的序列
        """
        #通过环境对象 env 的 og_env.scene.object_registry 方法获取场景中的笔和笔架对象
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # disturbance sequence
        pos0, orn0 = pen.get_position_orientation()
        orn0 = orn0.numpy()
        pose0 = np.concatenate([pos0, orn0])
        pos1 = pos0 + np.array([-0.08, 0.0, 0.0])
        orn1 = T.quat_multiply(T.euler2quat(np.array([0, 0, np.pi/4])), orn0)
        pose1 = np.concatenate([pos1, orn1])
        pos2 = pos1 + np.array([0.10, 0.0, 0.0])
        orn2 = T.quat_multiply(T.euler2quat(np.array([0, 0, -np.pi/2])), orn1)
        pose2 = np.concatenate([pos2, orn2])
        control_points = np.array([pose0, pose1, pose2])
        #使用样条插值生成一个包含 25 个步骤的干扰序列。
        pose_seq = spline_interpolate_poses(control_points, num_steps=25)
        def disturbance(counter):
            if counter < len(pose_seq):
                pose = pose_seq[counter]
                pos, orn = pose[:3], pose[3:]
                pen.set_position_orientation(pos, orn)
                counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1
    
    def stage2_disturbance_seq(env):
        """
        模拟在机器人试图重新定向笔时，将笔从夹具中取出的干扰序列
        Take the pen out of the gripper in stage 1 when robot is trying to reorient the pen
        """
        apply_disturbance = env.is_grasping()
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # disturbance sequence
        pos0, orn0 = pen.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pose1 = np.array([-0.30, -0.15, 0.71, -0.7071068, 0, 0, 0.7071068])
        control_points = np.array([pose0, pose1])
        pose_seq = spline_interpolate_poses(control_points, num_steps=25)
        def disturbance(counter):
            if apply_disturbance:
                if counter < 20:
                    if counter > 15:
                        env.robot.release_grasp_immediately()  # force robot to release the pen
                    else:
                        pass  # do nothing for the other steps
                elif counter < len(pose_seq) + 20:
                    env.robot.release_grasp_immediately()  # force robot to release the pen
                    pose = pose_seq[counter - 20]
                    pos, orn = pose[:3], pose[3:]
                    pen.set_position_orientation(pos, orn)
                    counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1
    
    def stage3_disturbance_seq(env):
        """
        Move the holder in stage 2 when robot is trying to drop the pen into the holder
        于模拟在机器人试图将笔放入笔架时，对笔架进行干扰的序列
        """
        pen = env.og_env.scene.object_registry("name", "pen_1")
        holder = env.og_env.scene.object_registry("name", "pencil_holder_1")
        # disturbance sequence
        pos0, orn0 = holder.get_position_orientation()
        pose0 = np.concatenate([pos0, orn0])
        pos1 = pos0 + np.array([-0.02, -0.15, 0.0])
        orn1 = orn0
        pose1 = np.concatenate([pos1, orn1])
        control_points = np.array([pose0, pose1])
        pose_seq = spline_interpolate_poses(control_points, num_steps=5)
        def disturbance(counter):
            if counter < len(pose_seq):
                pose = pose_seq[counter]
                pos, orn = pose[:3], pose[3:]
                holder.set_position_orientation(pos, orn)
                counter += 1
        counter = 0
        while True:
            yield disturbance(counter)
            counter += 1

    # task_list = {
    #     'pen': {
    #         'scene_file': './configs/og_scene_file_pen.json',
    #         'instruction': 'reorient the screwdrivere and drop it upright into the black pen holder',
    #         'rekep_program_dir': './vlm_query/pen',
    #         'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
    #         },
    # }
    # task = task_list['pen']
    # task_list = {
    #     'assembly': {
    #         'scene_file': './configs/og_scene_file_assembly_battery.json',
    #         'instruction': 'pick up the blue battery, do not rotate it, and install it on the green groove with initial orientation .',
    #         'rekep_program_dir': './vlm_query/pen',
    #         'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
    #         },
    # }
    # task = task_list['assembly']   
    # scene_file = task['scene_file']
    # instruction = task['instruction']
    # main = Main(scene_file, visualize=args.visualize)
    # main.perform_task(instruction,
    #                 rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None,
    #                 disturbance_seq=task.get('disturbance_seq', None) if args.apply_disturbance else None)
    task_list = {
        'assembly': {
            'scene_file': './configs/og_scene_file_assembly_battery_2.json',
            'instruction': 'Pick up the red cylinder, do not rotate it, and place it into the blue slot slowly.',
            'rekep_program_dir': './vlm_query/pen',
            'disturbance_seq': {1: stage1_disturbance_seq, 2: stage2_disturbance_seq, 3: stage3_disturbance_seq},
            },
    }
    task = task_list['assembly']   
    scene_file = task['scene_file']
    instruction = task['instruction']
    main = Main(scene_file, visualize=args.visualize)
    main.perform_task(instruction,
                    rekep_program_dir=task['rekep_program_dir'] if args.use_cached_query else None,
                    disturbance_seq=task.get('disturbance_seq', None) if args.apply_disturbance else None)