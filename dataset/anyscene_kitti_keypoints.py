import os.path as osp
import random
from typing import Optional

import glob
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
    back_project,
    render_with_z_buffer
)
from vision3d.utils.io import load_pickle, read_depth_image, read_image

def _get_frame_name(filename):
    _, seq_name, frame_name = filename.split(".")[0].split("/")
    seq_id = seq_name.split("-")[-1]
    frame_id = frame_name.split("_")[-1]
    output_name = f"{seq_id}-{frame_id}"
    return output_name

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

def posevec_T(pos):
    tvec = pos[0:3,0:1]
    rvec = pos[3:, 0:1]
    R = rvec.reshape((3,3))
    # tvec = np.matmul(R.T, -tvec)
    mat = np.eye(4)
    mat[:3,0:3] = R
    mat[:3,3:4] = tvec
    return mat

def visualize_depth_map(dep):
    valid_mask = (dep > 0)
    dep_min = np.min(dep[valid_mask])
    dep_max = np.max(dep[valid_mask])
    dep = (dep-dep_min)/(dep_max-dep_min)*255
    dep[~valid_mask] = 0
    dep = dep.astype(np.uint8)
    dep_vis = cv2.applyColorMap(dep, cv2.COLORMAP_JET)
    dep_vis[~valid_mask,:] = 0
    return dep_vis

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
        self.scene = "image/"
        self.data_dir = osp.join(self.dataset_dir, self.scene)
        self.subset = subset

        '''
            add other files in kitti
        '''
        self.sn_dir  = osp.join(self.dataset_dir, "normals/")
        self.int_dir = osp.join(self.dataset_dir, "intrinsics/")
        self.gt_dir  = osp.join(self.dataset_dir, "groundtruth_depth/")

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
            generate train and test list
        '''
        self.tmp_paths = glob.glob(self.data_dir + "*.png")
        self.tmp_paths.sort()
        self.int_paths = glob.glob(self.int_dir + "*.txt")
        self.int_paths.sort()
        self.gt_paths  = glob.glob(self.gt_dir  + "*.png")
        self.gt_paths .sort()
        self.sn_paths  = glob.glob(self.sn_dir  + "*.png")
        self.sn_paths .sort()
        num_file = int(len(self.tmp_paths))

        train_list, test_list = [], []
        int_train_list, int_test_list = [], []
        gt_train_list, gt_test_list = [], []
        sn_train_list, sn_test_list = [], []
        for i in range(num_file):
            data_idx = i
            if data_idx%2 == 0:
                train_list.append(data_idx)
                int_train_list.append(data_idx)
                gt_train_list.append(data_idx)
                sn_train_list.append(data_idx)
            else:
                test_list.append(data_idx)
                int_test_list.append(data_idx)
                gt_test_list.append(data_idx)
                sn_test_list.append(data_idx)
        
        if subset in ["val", "test"]:
            self.data_list = test_list
            self.int_list  = int_test_list
            self.gt_list   = gt_test_list
            self.sn_list   = sn_test_list
        else:
            self.data_list = train_list
            self.int_list  = int_train_list
            self.gt_list   = gt_train_list
            self.sn_list   = sn_train_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        data_dict = {}
        '''
            load data_index, intrinsics, image, and depth, camera pose
        '''
        data_index = self.data_list[index]
        data_dict["scene_name"] = self.scene
        data_dict["image_id"] = data_index
        data_dict["cloud_id"] = data_index
        # read image
        image_path = self.tmp_paths[data_index]
        image = cv2.imread(image_path)/ 255.0
        h_old, w  = image.shape[0], image.shape[1]
        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        # read depth
        '''
            a trick to minimize the depth from [0,20]m to [0,4m]
            because kpconv and knn parameter only support near-depth scene

            depth has the scale of 1000.0
        '''
        depth_scale = 5
        depth_path = self.gt_paths[data_index]
        depth = read_depth_image(depth_path, depth_scale).astype(np.float)

        '''
            a trick to resize from 1216*356 to 640*480 
            1216/2 = 608
        '''
        image_re = np.zeros((480,640,3), dtype=np.float32)
        depth_re = np.zeros((480,640), dtype=np.float32)
        image_re[0:h_old,0:640,0:3] = image[0:h_old,(608-320):(608+320),0:3]
        depth_re[0:h_old,0:640]     = depth[0:h_old,(608-320):(608+320)]
        image = image_re
        depth = depth_re
        h, w  = image.shape[0], image.shape[1]
        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        # read 2d surface normal
        rw_sn_path = self.sn_paths[data_index]
        rw_sn_image = cv2.imread(rw_sn_path)
        sn_map = rw_sn_image.astype(np.float64)
        sn_map = sn_map/255.0*2 - 1.0
        # align surface normal to point cloud normal (mostly aligned)
        mask = ((sn_map[:,:,2] < 0) & (sn_map[:,:,1] < 0)) & (sn_map[:,:,0] > 0)
        sn_map[mask,:] *= (-1)
        image_re = np.zeros((480,640,3), dtype=np.float32)
        image_re[0:h_old,0:640,0:3] = sn_map[0:h_old,(608-320):(608+320),0:3]
        sn_map = image_re

        # generate 2d keypoints (candidate) and keypoints mask
        _path = image_path
        rw_image_path = _path[:-4] + ".png"
        rw_image = cv2.imread(rw_image_path)
        image_re = np.zeros((480,640,3), dtype=np.uint8)
        image_re[0:h_old,0:640,0:3] = rw_image[0:h_old,(608-320):(608+320),0:3]
        rw_image = image_re
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

        '''
            a patch to add missing parts
        '''
        data_dict["image_file"] = image_path
        data_dict["depth_file"] = depth_path

        # read intrinsic matrix
        int_path = self.int_paths[data_index]
        int_data = np.loadtxt(int_path)
        intrinsics = np.zeros((3,3))
        intrinsics[2,2] = 1.0
        intrinsics[0,0] = int_data[0]
        intrinsics[1,1] = int_data[4]
        intrinsics[0,2] = 640/2 #int_data[2]
        intrinsics[1,2] = 480/2 #int_data[5]
        # intrinsics[0,2] = int_data[2]
        # intrinsics[1,2] = int_data[5]
        
        # get pose
        transform = np.eye(4)

        # read points with down-sampling
        depth_limit = 25.0/depth_scale
        points_mat = back_project(depth, intrinsics, depth_limit=depth_limit, return_matrix=True)
        valid_map = [points_mat[:,:,2] > 0]
        points = points_mat[valid_map] # [N,3]
        points_rgb = image[valid_map]

        points_normals = sn_map[valid_map]
        points_raw = points_mat[valid_map]
        points_rgb = image[valid_map]
        points_mask_1x = mask_1x[valid_map]
        points_mask_3x = mask_3x[valid_map]
        points_mask_5x = mask_5x[valid_map]
        points_mask = np.concatenate((
            points_mask_1x[:,0:1], points_mask_3x[:,0:1], points_mask_5x[:,0:1]
            ), axis=1)/255.0

        sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
        if self.max_points is not None and points.shape[0] > self.max_points:
            points = points[sel_indices]
            points_rgb = points_rgb[sel_indices]
            points_mask = points_mask[sel_indices]
            points_normals = points_normals[sel_indices]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(points_rgb[:,:3])
        # pcd = pcd.voxel_down_sample(voxel_size=0.05)
        points = np.array(pcd.points)
        points_rgb = np.array(pcd.colors)

        '''
            check the visulization and data --- ok
        '''
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # print("data_index: ", data_index)
        # print("max depth: ", np.max(depth))
        # print("intrinsics:  ", intrinsics)
        # print("points: ", points.shape)
        # show_pcd(pcd)
        # vis_depth = visualize_depth_map(depth)
        # cv2.imshow("vis_depth", vis_depth)
        # cv2.waitKey(0)
        # assert 1==-1

        '''
            check the projection results --- ok
        '''
        # pixels, masks, depths = render_with_z_buffer(points, intrinsics, h, w, transform, True)
        # pixels = pixels[masks]
        # depths = depths[masks]
        # points_rgb = points_rgb[masks]
        # img_render = image-image #np.zeros((h,w,3), dtype=np.uint8)
        # for i in range(pixels.shape[0]):
        #     u = int(pixels[i,0])
        #     v = int(pixels[i,1])
        #     img_render[u,v,0] = points_rgb[i,0]*255
        #     img_render[u,v,1] = points_rgb[i,1]*255
        #     img_render[u,v,2] = points_rgb[i,2]*255
        # img_render = img_render.astype(np.uint8)
        # cv2.imshow("img_render", img_render)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
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
                # this
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_mutual(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d)
            else:
                img_corr_pixels, pcd_corr_indices = get_2d3d_correspondences_radius(
                    depth, points, intrinsics, transform, self.matching_radius_2d, self.matching_radius_3d)
            img_corr_indices = img_corr_pixels[:, 0] * image.shape[1] + img_corr_pixels[:, 1]
            data_dict["img_corr_pixels"] = img_corr_pixels
            data_dict["img_corr_indices"] = img_corr_indices
            data_dict["pcd_corr_indices"] = pcd_corr_indices

            '''
                check the gt correspondences in kitti dataset --- ok
            '''
            # print("img_corr_pixels: ", img_corr_pixels.shape)
            # assert 1==-1

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

