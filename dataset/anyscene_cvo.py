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
        self.scene = "scene_00/"
        self.data_dir = osp.join(self.dataset_dir, self.scene)
        self.subset = subset

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
        tmp_paths = glob.glob(self.data_dir + "*.npy")
        tmp_paths.sort()
        num_file = int(len(tmp_paths)/3)-1

        train_list, test_list = [], []
        for i in range(num_file):
            data_idx = i+1
            if data_idx%2 == 0:
                train_list.append(data_idx)
            else:
                test_list.append(data_idx)
        
        if subset in ["val", "test"]:
            self.data_list = test_list
        else:
            self.data_list = train_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        data_dict = {}
        '''
            load data_index, intrinsics, image, and depth, camera pose
        '''
        data_index = self.data_list[index]
        data_index = 1147 #1
        data_index_ref = int(data_index/10)*10 + 1
        data_dict["scene_name"] = self.scene
        data_dict["image_id"] = data_index
        data_dict["cloud_id"] = data_index_ref

        intrinsics = np.zeros((3,3))
        intrinsics[2,2] = 1.0
        intrinsics[0,0] = 863.4241/2
        intrinsics[1,1] = 863.4171/2
        intrinsics[0,2] = 640.6808/2
        intrinsics[1,2] = 518.3392/2

        # read image
        image = cv2.imread(self.data_dir+str(data_index)+".png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/ 255.0
        image_ref = cv2.imread(self.data_dir+str(data_index_ref)+".png")
        image_ref = cv2.cvtColor(image_ref, cv2.COLOR_BGR2RGB)/ 255.0
        h, w  = image.shape[0], image.shape[1]
        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]
        depth = np.fromfile(self.data_dir+str(data_index_ref)+"_depth.bin", dtype=np.float32)
        depth = depth.reshape((h,w))

        pose = np.load(self.data_dir+str(data_index)+".npy")
        pose_ref = np.load(self.data_dir+str(data_index_ref)+".npy")
        pose = posevec_T(pose)
        pose_ref = posevec_T(pose_ref)

        # read points with down-sampling
        '''
            directly using point cloud back-projected from depth image xyz+rgb
        '''
        depth_limit = 50.0
        points_mat = back_project(depth, intrinsics, scaling_factor=1, depth_limit=depth_limit, return_matrix=True)
        valid_map = [points_mat[:,:,2] > 0]
        points = points_mat[valid_map] # [N,3]
        points_rgb = image_ref[valid_map]

        # convert camera coordinate system to world coordinate system
        R_ref = pose_ref[0:3,0:3]
        t_ref = pose_ref[0:3,3:4]
        points_tmp = points.T # [3,N]
        points_tmp = np.matmul(R_ref, points_tmp)
        points_tmp = points_tmp + t_ref
        points_tmp = points_tmp.T

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_tmp[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(points_rgb[:,:3])
        # pcd = pcd.voxel_down_sample(voxel_size=0.05)
        points = np.array(pcd.points)
        points_rgb = np.array(pcd.colors)

        # projection again
        pose_curr = np.linalg.inv(pose)
        pose_curr = np.linalg.inv(pose_ref)
        R_curr = pose_curr[0:3,0:3]
        t_curr = pose_curr[0:3,3:4]
        # points_tmp = points.T # [3,N]
        # points_tmp = np.matmul(R_ref, points_tmp)
        # points_tmp = points_tmp + t_ref
        pixels, masks, depths = render_with_z_buffer(points, intrinsics, h, w, pose_curr, True)
        pixels = pixels[masks]
        depths = depths[masks]
        points_rgb = points_rgb[masks]
        img_render = image-image #np.zeros((h,w,3), dtype=np.uint8)
        for i in range(pixels.shape[0]):
            u = int(pixels[i,0])
            v = int(pixels[i,1])
            img_render[u,v,0] = points_rgb[i,0]*255
            img_render[u,v,1] = points_rgb[i,1]*255
            img_render[u,v,2] = points_rgb[i,2]*255
        print("===> ", np.max(img_render))
        img_render = img_render.astype(np.uint8)

        print("data_index: ", data_index)
        print("data_index_ref: ", data_index_ref)
        print("h, w: ", h, w)
        print("depth: ", depth.shape, np.max(depth))
        print("points: ", points.shape)
        show_pcd(pcd)
        cv2.imshow("img: ", image)
        cv2.imshow("img_render: ", img_render)
        cv2.waitKey(0)
        assert 1==-1

        sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
        if self.max_points is not None and points.shape[0] > self.max_points:
            points = points[sel_indices]
            points_rgb = points_rgb[sel_indices]

        '''
            transformation correction as identity matrix
        '''
        transform = np.eye(4)

        '''
            visulization of colorful point cloud
        '''
        # print("points: ", points.shape)
        # show_pcd(pcd)
        # print("transform: ", transform)
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
        return data_dict

