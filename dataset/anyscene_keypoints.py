import os.path as osp
import random
from typing import Optional

import cv2
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset

from vision3d.array_ops import (
    apply_transform,
    compose_transforms,
    get_2d3d_correspondences_mutual,
    get_2d3d_correspondences_radius,
    get_transform_from_rotation_translation,
    inverse_transform,
    random_sample_small_transform,
    back_project
)
from vision3d.utils.io import load_pickle, read_depth_image, read_image

def get_sobel_res(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x) # 转回unit8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst

def vis_depth(dep):
    min_dep = np.min(dep)
    max_dep = np.max(dep)
    tmp = (dep-min_dep)/(max_dep-min_dep)*255
    tmp = tmp.astype(np.uint8)
    tmp = cv2.applyColorMap(tmp, cv2.COLORMAP_JET)
    return tmp

def show_pcd(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window("point cloud")
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    render_options.background_color = np.array([0,0,0])
    render_options.point_size = 3.0
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run() 

class I2PHardPairDataset(Dataset):
    '''
        anpei-note 1104
        in the original matr dataset split, most of point clouds are sparse
            which is not consistent in the real ar applications
    '''
    def __init__(
        self,
        dataset_dir: str,
        subset: str,
        max_points: Optional[int] = None,
        return_corr_indices: bool = False,
        matching_method: str = "mutual_nearest",
        matching_radius_2d: float = 8.0,
        matching_radius_3d: float = 0.0375,
        scene_name: Optional[str] = None,
        overlap_threshold: Optional[float] = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        scale_augmentation: bool = False,
        return_overlap_indices: bool = False):

        super().__init__()
        assert subset in ["trainval", "train", "val", "test"]
        assert matching_method in ["mutual_nearest", "radius"], f"Bad matching method: {matching_method}"

        self.dataset_dir = dataset_dir
        self.data_dir = osp.join(self.dataset_dir, "data")
        self.metadata_dir = osp.join(self.dataset_dir, "metadata")
        self.subset = subset
        # self.subset = "train" # it is used only for kitchen->rgbd experiment setting
        self.metadata_list = load_pickle(osp.join(self.metadata_dir, f"{self.subset}-full.pkl"))

        if scene_name is not None:
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] == scene_name]

        self.max_points = max_points
        self.return_corr_indices = return_corr_indices
        self.matching_method = matching_method
        self.matching_radius_2d = matching_radius_2d
        self.matching_radius_3d = matching_radius_3d
        self.overlap_threshold = overlap_threshold
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.scale_augmentation = scale_augmentation
        self.return_overlap_indices = return_overlap_indices

        '''
            using different ratio (i.e. 5%, 10%, 20%) for train or eval
        '''
        is_use_train_ratio = True
        # is_use_train_ratio = False
        metadata_list_new = []
        if is_use_train_ratio:
            num_all = len(self.metadata_list)
            for i in range(num_all):
                if i%20 == 0:
                # if i%5 == 0:
                # if i%1 == 0:
                    metadata_list_new.append(self.metadata_list[i])
            self.metadata_list = metadata_list_new
        
        '''
            using specific scene for training or eval
                which is used for domain generalization
        '''
        is_use_scene_train = True
        if subset in ["val", "test"]:
            is_use_scene_train = False
            # is_use_scene_train = True
        if is_use_scene_train:
            scene_spec_name = "chess"
            # scene_spec_name = "heads"
            scene_spec_name = "office"
            # scene_spec_name = "redkitchen"
            # scene_spec_name = "pumpkin"
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] == scene_spec_name]

    def __len__(self):
        return len(self.metadata_list)

    def __getitem__(self, index: int):
        data_dict = {}

        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["image_file"] = metadata["image_file"]
        data_dict["depth_file"] = metadata["depth_file"]
        data_dict["image_id"] = index
        data_dict["cloud_id"] = index

        intrinsics_file = osp.join(self.data_dir, metadata["scene_name"], "camera-intrinsics.txt")
        intrinsics = np.loadtxt(intrinsics_file)

        # read image
        depth = read_depth_image(osp.join(self.data_dir, metadata["depth_file"])).astype(np.float)
        image = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=False)
        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        # read 2d surface normal
        # pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)
        _path = str(osp.join(self.data_dir, metadata["image_file"]))
        rw_sn_path = _path[:-4] + "_dsine.png" 
        rw_sn_image = cv2.imread(rw_sn_path)
        sn_map = rw_sn_image.astype(np.float64)
        sn_map = sn_map/255.0*2 - 1.0
        # align surface normal to point cloud normal (mostly aligned)
        mask = ((sn_map[:,:,2] < 0) & (sn_map[:,:,1] < 0)) & (sn_map[:,:,0] > 0)
        sn_map[mask,:] *= (-1)

        # generate 2d keypoints (candidate) and keypoints mask
        _path = str(osp.join(self.data_dir, metadata["image_file"]))
        rw_image_path = _path[:-4] + ".png"
        rw_image = cv2.imread(rw_image_path)
        rw_image_gray = cv2.cvtColor(rw_image,cv2.COLOR_BGR2GRAY)
        corners  = cv2.goodFeaturesToTrack(rw_image_gray,4096,0.001,1)
        mask_0x  = np.zeros_like(rw_image)
        mask_1x  = np.zeros_like(rw_image)
        mask_3x  = np.zeros_like(rw_image)
        mask_5x  = np.zeros_like(rw_image)
        kpts_2d_pixels = np.zeros((len(corners),2), dtype=np.float16)
        cnt = 0
        for i in corners:
            x,y = i.ravel()
            kpts_2d_pixels[cnt, 0] = y
            kpts_2d_pixels[cnt, 1] = x
            mask_0x[int(y), int(x),0] = 255
            cv2.circle(rw_image, (int(x), int(y)), radius=3, color=(255,0,0), thickness=2)
            cv2.circle(mask_1x,  (int(x), int(y)), radius=1, color=(255,0,0), thickness=-1)
            cv2.circle(mask_3x,  (int(x), int(y)), radius=3, color=(255,0,0), thickness=-1)
            cv2.circle(mask_5x,  (int(x), int(y)), radius=5, color=(255,0,0), thickness=-1)
            cnt += 1
        images_mask = mask_0x[:,:,0:1]/255.0

        # read points with down-sampling
        '''
            directly using point cloud back-projected from depth image xyz+rgb
        '''
        depth_limit = 6.0
        points_mat = back_project(depth, intrinsics, depth_limit=depth_limit, return_matrix=True)
        valid_map = np.array(points_mat[:,:,2] > 0)
        points_raw = points_mat[valid_map]
        points_rgb = image[valid_map]
        points_mask_1x = mask_1x[valid_map]
        points_mask_3x = mask_3x[valid_map]
        points_mask_5x = mask_5x[valid_map]
        points_mask = np.concatenate((
            points_mask_1x[:,0:1], points_mask_3x[:,0:1], points_mask_5x[:,0:1]
            ), axis=1)/255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_raw[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(points_rgb[:,:3])
        pcd = pcd.voxel_down_sample(voxel_size=0.015) # 0.015
        points      = np.array(pcd.points)
        points_rgb  = np.array(pcd.colors)

        # compute 3d surface normal --- ok
        '''
            a dataloader bug, please compute the point cloud surface normal before
        '''
        # pcd.estimate_normals(
        #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        # points_normals = np.array(pcd.normals)
        # mask = (points_normals[:,2] <= 0)
        # points_normals[mask,:] *= (-1)
        # pcd.normals = o3d.utility.Vector3dVector(points_normals[:,:3])
        # o3d.visualization.draw_geometries([pcd],
        #     point_show_normal=True)
        # assert 1==-1

        # test 2d surface normal estimation --- mostly ok
        points_normal = sn_map[valid_map]
        pcd_test = o3d.geometry.PointCloud()
        pcd_test.points = o3d.utility.Vector3dVector(points_raw[:,:3])
        pcd_test.colors = o3d.utility.Vector3dVector(points_rgb[:,:3])
        pcd_test.normals = o3d.utility.Vector3dVector(points_normal[:,:3])
        pcd_test = pcd_test.voxel_down_sample(voxel_size=0.015)
        points_normals = np.array(pcd_test.normals)
        # o3d.visualization.draw_geometries([pcd_test],
        #     point_show_normal=True)
        # assert 1==-1

        pcd.points = o3d.utility.Vector3dVector(points_raw[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(points_mask[:,:3])
        pcd = pcd.voxel_down_sample(voxel_size=0.015) # 0.015
        points_mask = np.array(pcd.colors)
        # print("points_mask: ", points_mask)

        sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
        if self.max_points is not None and points.shape[0] > self.max_points:
            points = points[sel_indices]
            points_rgb  = points_rgb[sel_indices]
            points_mask = points_mask[sel_indices]
            points_normals = points_normals[sel_indices]

        '''
            transformation correction as identity matrix
        '''
        transform = np.eye(4)

        '''
            visulization of colorful point cloud
        '''
        # for i in corners:
        #     x,y = i.ravel()
        #     cv2.circle(rw_image, (int(x), int(y)), radius=1, color=255, thickness=-1)
        # # print("points: ", points.shape)
        # cv2.imshow("rw_image", rw_image)
        # cv2.imshow("mask_1x", mask_1x)
        # cv2.imshow("mask_5x", mask_5x)
        # show_pcd(pcd)
        # cv2.waitKey()
        # # print("transform: ", transform)
        # assert 1==-1

        if self.use_augmentation:
            # augment point cloud
            aug_transform = random_sample_small_transform()
            center = points.mean(axis=0)
            subtract_center = get_transform_from_rotation_translation(None, -center)
            add_center = get_transform_from_rotation_translation(None, center)
            aug_transform = compose_transforms(subtract_center, aug_transform, add_center)
            points = apply_transform(points, aug_transform)
            inv_aug_transform = inverse_transform(aug_transform)
            transform = compose_transforms(inv_aug_transform, transform)
            points += (np.random.rand(points.shape[0], 3) - 0.5) * self.aug_noise

        if self.scale_augmentation and random.random() > 0.5:
            # augment image
            scale = random.uniform(1.0, 1.2)
            raw_image_h = image.shape[0]
            raw_image_w = image.shape[1]
            new_image_h = int(raw_image_h * scale)
            new_image_w = int(raw_image_w * scale)
            start_h = new_image_h // 2 - raw_image_h // 2
            end_h = start_h + raw_image_h
            start_w = new_image_w // 2 - raw_image_w // 2
            end_w = start_w + raw_image_w
            image = cv2.resize(image, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            image = image[start_h:end_h, start_w:end_w]
            depth = cv2.resize(depth, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            depth = depth[start_h:end_h, start_w:end_w]
            intrinsics[0, 0] = intrinsics[0, 0] * scale
            intrinsics[1, 1] = intrinsics[1, 1] * scale

        # build correspondences
        if self.return_corr_indices:
            if self.matching_method == "mutual_nearest":
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_mutual(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d)
            else:
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_radius(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d)
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[1] + img_corr_pixels[:, 1]
            data_dict["img_corr_pixels"] = img_corr_pixels
            data_dict["img_corr_indices"] = img_corr_indices
            data_dict["pcd_corr_indices"] = pcd_corr_indices

        if self.return_overlap_indices:
            img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_radius(
                depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d)
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[1] + img_corr_pixels[:, 1]
            img_overlap_indices = np.unique(img_corr_indices)
            pcd_overlap_indices = np.unique(pcd_corr_indices)
            img_overlap_h_pixels = img_overlap_indices // image.shape[1]
            img_overlap_w_pixels = img_overlap_indices % image.shape[1]
            img_overlap_pixels = np.stack([img_overlap_h_pixels, img_overlap_w_pixels], axis=1)
            data_dict["img_overlap_pixels"] = img_overlap_pixels
            data_dict["img_overlap_indices"] = img_overlap_indices
            data_dict["pcd_overlap_indices"] = pcd_overlap_indices
        
        # build data dict
        data_dict["intrinsics"] = intrinsics.astype(np.float32)
        data_dict["transform"] = transform.astype(np.float32)
        data_dict["image"] = image.astype(np.float32)
        data_dict["depth"] = depth.astype(np.float32)
        data_dict["points"] = points.astype(np.float32)
        data_dict["points_rgb"] = points_rgb.astype(np.float32)
        data_dict["points_mask"] = points_mask.astype(np.float32)
        data_dict["images_mask"] = images_mask.astype(np.float32)
        data_dict["kpts_2d_pixels"] = kpts_2d_pixels.astype(np.float32)
        data_dict["surface_normal_2d"] = sn_map.astype(np.float32)
        data_dict["surface_normal_3d"] = points_normals.astype(np.float32)
        return data_dict 

