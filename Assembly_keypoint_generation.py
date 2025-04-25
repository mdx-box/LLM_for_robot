import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from utils import filter_points_by_bounds
from sklearn.cluster import MeanShift
from utils import get_config
#该代码的主要功能是从输入的RGB图像中提取具有代表性的关键点。这些关键点是通过DINO v2提取的特征向量，并在特征空间和笛卡尔空间中进行聚类和合并，
# 从而生成对图像中不同区域的代表性关键点。这些关键点可以用于进一步的分析或图像处理应用，例如目标检测或图像分割。
class KeypointProposer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        #关键点跟踪
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(self.device)
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.mean_shift = MeanShift(bandwidth=self.config['min_dist_bt_keypoints'], bin_seeding=True, n_jobs=32)
        self.patch_size = 14  # dinov2
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])

    def get_keypoints(self, rgb, points, masks):
        # preprocessing 图像处理，使其符合dinv2的标准
        transformed_rgb, rgb, points, masks, shape_info = self._preprocess(rgb, points, masks)
        # get features 返回的shape为[H*W, feature_dim]
        features_flat = self._get_features(transformed_rgb, shape_info)
        # for each mask, cluster in feature space to get meaningful regions, and uske their centers as keypoint candidates
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(points, features_flat, masks)
        # exclude keypoints that are outside of the workspace
        within_space = filter_points_by_bounds(candidate_keypoints, self.bounds_min, self.bounds_max, strict=True)
        candidate_keypoints = candidate_keypoints[within_space]
        candidate_pixels = candidate_pixels[within_space]
        candidate_rigid_group_ids = candidate_rigid_group_ids[within_space]
        # merge close points by clustering in cartesian space
        merged_indices = self._merge_clusters(candidate_keypoints)
        candidate_keypoints = candidate_keypoints[merged_indices]
        candidate_pixels = candidate_pixels[merged_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[merged_indices]
        # sort candidates by locations
        sort_idx = np.lexsort((candidate_pixels[:, 0], candidate_pixels[:, 1]))
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]
        # project keypoints to image space
        projected = self._project_keypoints_to_img(rgb.cpu().numpy(), candidate_pixels, candidate_rigid_group_ids, masks, features_flat)
        return candidate_keypoints, projected

    def _preprocess(self, rgb, points, masks):
        #对图像进行处理，使其符合dinov2的标准
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        # convert masks to binary masks
        masks = [masks == uid for uid in np.unique(masks)]
        # ensure input shape is compatible with dinov2
        H, W, _ = rgb.shape
        patch_h = int(H // self.patch_size)
        patch_w = int(W // self.patch_size)
        new_H = patch_h * self.patch_size
        new_W = patch_w * self.patch_size
        transformed_rgb = cv2.resize(rgb.cpu().numpy(), (new_W, new_H))
        transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # float32 [H, W, 3]
        # shape info
        shape_info = {
            'img_h': H,
            'img_w': W,
            'patch_h': patch_h,
            'patch_w': patch_w,
        }
        return transformed_rgb, rgb, points, masks, shape_info
    
    def _project_keypoints_to_img(self, rgb, candidate_pixels, candidate_rigid_group_ids, masks, features_flat):
        #该方法用于将关键点投影到图像上，并在图像上绘制矩形框和标记关键点序号。
        if isinstance(rgb, torch.Tensor):
            projected = rgb.clone().cpu().numpy()
        else:
            projected = rgb.copy()
        # overlay keypoints on the image
        for keypoint_count, pixel in enumerate(candidate_pixels):
            #这里遍历每个候选关键点，keypoint_count 是当前关键点的序号，pixel 是关键点在图像上的像素坐标。
            #这段代码首先生成要显示的文本 displayed_text，然后计算文本的长度 text_length。根据文本长度计算矩形框的宽度 box_width 和高度 box_height，
            # 然后使用 OpenCV 的 rectangle 函数在图像上绘制矩形框。第一个矩形框是白色填充的，第二个矩形框是黑色边框。
            displayed_text = f"{keypoint_count}"
            text_length = len(displayed_text)
            # draw a box
            box_width = 30 + 10 * (text_length - 1)
            box_height = 30
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
            cv2.rectangle(projected, (pixel[1] - box_width // 2, pixel[0] - box_height // 2), (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
            # draw text
            org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
            color = (255, 0, 0)
            cv2.putText(projected, str(keypoint_count), org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            keypoint_count += 1
        return projected

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):
        #提取图像特征（调用 _get_features 方法）。
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']
        # get features
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)  # float32 [1, 3, H, W]
        assert img_tensors.shape[1] == 3, "unexpected image shape"
        #将图像输入到 dinov2 模型中，获取特征网格。
        features_dict = self.dinov2.forward_features(img_tensors)
        #从提取结果中获取x_norm_patchtokens
        raw_feature_grid = features_dict['x_norm_patchtokens']  # float32 [num_cams, patch_h*patch_w, feature_dim]
        #将其reshape为（1， patch_h, patch_w， feature_dim)
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w, -1)  # float32 [num_cams, patch_h, patch_w, feature_dim]
        # compute per-point feature using bilinear interpolation
        interpolated_feature_grid = interpolate(raw_feature_grid.permute(0, 3, 1, 2),  # float32 [num_cams, feature_dim, patch_h, patch_w]
                                                size=(img_h, img_w),#指定输出图像的size
                                                mode='bilinear').permute(0, 2, 3, 1).squeeze(0)  # float32 [H, W, feature_dim] #插值方式为线性插值
        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])  # float32 [H*W, feature_dim] #将图像reshape为(H*W, feature_dim)
        return features_flat #返回的shape为(H*W, feature_dim)

    def _cluster_features(self, points, features_flat, masks):
        #在每个掩码区域内聚类特征，以生成候选关键点。这段代码通过特征提取、插值和聚类等步骤，从输入的 RGB 图像中生成了具有代表性的关键点，这些关键点可以用于进一步的图像处理和分析
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []
        for rigid_group_id, binary_mask in enumerate(masks):
            # ignore mask that is too large 忽略太大的mask
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue
            # consider only foreground features 只考虑最显著的位置
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            #返回binary_mask中不是的索引值
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]
            # reduce dimensionality to be less sensitive to noise and texture
            obj_features_flat = obj_features_flat.double()
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            features_pca = (features_pca - features_pca.min(0)[0]) / (features_pca.max(0)[0] - features_pca.min(0)[0])
            X = features_pca
            # add feature_pixels as extra dimensions
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            feature_points_torch  = (feature_points_torch - feature_points_torch.min(0)[0]) / (feature_points_torch.max(0)[0] - feature_points_torch.min(0)[0])
            X = torch.cat([X, feature_points_torch], dim=-1)
            # cluster features to get meaningful regions使用 kmeans 聚类方法对特征进行聚类，得到每个特征点所属的簇和簇中心
            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config['num_candidates_per_mask'],
                distance='euclidean',
                device=self.device,
            )
            cluster_centers = cluster_centers.to(self.device)
            for cluster_id in range(self.config['num_candidates_per_mask']):
                cluster_center = cluster_centers[cluster_id][:3]
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]
                member_features = features_pca[member_idx]
                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])
                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        #使用 MeanShift 聚类方法合并靠近的关键点
        #初始化聚类器
        self.mean_shift.fit(candidate_keypoints)
        #获取聚类中信
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        #遍历聚类中心
        for center in cluster_centers:
            #计算每个候选关键点到当前聚类中心的欧几里得距离，并将最小距离的索引添加到 merged_indices 列表中。
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices


if __name__ == "__main__":
    import yaml
    import os
    
    # 加载配置文件
    config = get_config(config_path="./configs/config.yaml")
    
    # 初始化KeypointProposer
    proposer = KeypointProposer(config["keypoint_proposer"])
    
    # 读取本地图像
    image_path = "camera_data/rgb_image.jpg"  # 确保图像文件存在
    rgb = cv2.imread(image_path)
    print(f'rgb:{rgb}')
    print(f'type of rgb:{type(rgb)}')
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # 创建虚拟的点云和掩码数据
    points =  np.load('camera_data/points.npy')
    masks = np.load('camera_data/mask.npy')
    # 生成关键点和投影图像
    keypoints, projected_img = proposer.get_keypoints(torch.from_numpy(rgb), points, masks)
    
    # 保存结果
    output_path = "keypoints_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(projected_img, cv2.COLOR_RGB2BGR))
    print(f"关键点已生成并保存到 {output_path}")
    print(f"检测到的关键点数量: {len(keypoints)}")
    print("关键点坐标:")
    for i, kp in enumerate(keypoints):
        print(f"关键点 {i}: {kp}")