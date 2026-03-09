import os.path as osp
import random
from typing import Optional

import glob
import copy
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

import trimesh
import csv
import glob
import os
from PIL import Image

class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            self.poses += [np.linalg.inv(T)]

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)

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
        self.scene = ""
        self.data_dir = osp.join(self.dataset_dir, self.scene)
        self.subset = subset
        self.tumparser = TUMParser(self.data_dir)

        self.data_list = []
        for i in range(self.tumparser.n_img):
            if i%10 == 0: # 5 10
                self.data_list.append(i)

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
            generate pixel arrays 
        '''
        self.pixel_coords = np.zeros((480, 640, 2))
        for u in range(640):
            for v in range(480):
                self.pixel_coords[v,u,0] = u #(u-cx)/fx
                self.pixel_coords[v,u,1] = v #(v-cy)/fy

    def analyze_sam_2d(self, seg_img):
        sa = (seg_img * 255).astype(np.uint8) # [H,W,3]
        sa_v = sa[:,:,0]*1e6 + sa[:,:,1]*1e3 + sa[:,:,2]
        i_a, i_b = np.unique(sa_v, return_inverse=True)
        num_pts_seg = len(i_a)

        seg_2d_id = []
        seg_2d_cc = []
        seg_2d_mask = []
        seg_2d_ct = []

        pixel_coords = copy.deepcopy(self.pixel_coords)
        for i in range(num_pts_seg):
            # removal un-segmented area
            if i_a[i] == 0:
                continue
            mask_i = (i_b == i).reshape(seg_img.shape[0], seg_img.shape[1])
            coords = pixel_coords[mask_i]
            # mean_coords = np.median(coords, axis=0).astype(np.int)
            # cc_i = sa_v[mean_coords[1], mean_coords[0]]
            cc_i = i_a[i]
            ct_i = np.mean(coords, axis=0)
            
            seg_2d_id.append(i)
            seg_2d_cc.append(cc_i)
            seg_2d_mask.append(mask_i)
            seg_2d_ct.append(ct_i)
        return seg_2d_id, seg_2d_cc, seg_2d_mask, seg_2d_ct
    
    def analyze_sam_3d(self, pts, seg_pts):
        sb = (seg_pts * 255).astype(np.uint8) # [N,3]
        sb_v = sb[:,0]*1e6 + sb[:,1]*1e3 + sb[:,2]
        p_a, p_b  = np.unique(sb_v, return_inverse=True)
        num_pcd_seg = len(p_a)

        seg_3d_id = []
        seg_3d_cc = []
        seg_3d_mask = []
        seg_3d_ct = []

        for i in range(num_pcd_seg):
            # removal un-segmented area
            if p_a[i] == 0:
                continue
            mask_i = (p_b == i)
            coords = pts[mask_i]
            cc_i = p_a[i]
            ct_i = np.mean(coords, axis=0) # median, mean

            seg_3d_id.append(i)
            seg_3d_cc.append(cc_i)
            seg_3d_mask.append(mask_i)
            seg_3d_ct.append(ct_i)
        return seg_3d_id, seg_3d_cc, seg_3d_mask, seg_3d_ct
    
    def align_sam_2d_3d(self, seg_2d_id, seg_3d_id, seg_2d_cc, seg_3d_cc):
        num_2d = len(seg_2d_id)
        num_3d = len(seg_3d_id)
        align_list = np.zeros((num_2d, 2))
        mat_primitive = np.zeros((num_2d, num_3d))
        for i in range(num_2d):
            for j in range(num_3d):
                d = np.abs(seg_2d_cc[i] - seg_3d_cc[j])
                if d <= 1:
                    align_list[i,0] = i
                    align_list[i,1] = j
                    mat_primitive[i,j] = 1
                    continue
        return align_list, mat_primitive

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index: int):
        data_dict = {}
        '''
            load data_index, intrinsics, image, and depth, camera pose

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }
            self.frames.append(frame)
        '''
        data_index = self.data_list[index]
        data_dict["scene_name"] = self.scene
        data_dict["image_id"] = data_index
        data_dict["cloud_id"] = data_index
        data_info = self.tumparser.frames[data_index]

        # read image
        image_path = data_info["file_path"]
        image = cv2.imread(image_path)/ 255.0
        h, w  = image.shape[0], image.shape[1]
        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]
        pixel_coords = self.pixel_coords # [H,W,3]

        # read depth
        depth_scale = 4.0 # especially for scene_03
        depth_path = data_info["depth_path"]
        depth = read_depth_image(depth_path, depth_scale).astype(np.float)

        h, w  = image.shape[0], image.shape[1]
        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]
        data_dict["image_file"] = image_path
        data_dict["depth_file"] = depth_path

        '''
            anpei add load segment anything model
        '''
        seg_path = image_path.replace("/rgb/", "/sam/")
        seg_path = seg_path[:-4] + "_sampd.png"
        seg_img_a = cv2.imread(seg_path)/ 255.0
        seg_img_b = cv2.imread(seg_path)/ 255.0

        '''
            read depth predicted by depthanything v2 (metric-depth) --- ok
        '''
        is_need_depth_v2 = False
        is_need_depth_v2 = True
        if is_need_depth_v2:
            depth_path_v2 = image_path[:-4] + "_depth_v2.bin"
            depth_path_v2 = image_path[:-4] + "_depth_v2_tiny.bin" # using the different mde models
            depth_pred = np.fromfile(depth_path_v2, np.int16)
            depth_pred = depth_pred.reshape((image.shape[0], image.shape[1]))
            # compute the scalar factor per pixels
            ratio_mat  = (depth_pred)/depth/100
            invalid_mask = (depth <= 1)
            ratio_mat[invalid_mask] = 0

        # read intrinsic matrix
        intrinsics = np.zeros((3,3))
        intrinsics[2,2] = 1.0
        # ==> scene_01
        # intrinsics[0,0] = 517.306408
        # intrinsics[1,1] = 516.469215
        # intrinsics[0,2] = 318.643040
        # intrinsics[1,2] = 255.313989
        # ==> scene_02
        intrinsics[0,0] = 520.90862
        intrinsics[1,1] = 521.007327
        intrinsics[0,2] = 325.141442
        intrinsics[1,2] = 249.701764
        # # ==> scene_03
        # intrinsics[0,0] = 535.4
        # intrinsics[1,1] = 539.2
        # intrinsics[0,2] = 320.1
        # intrinsics[1,2] = 247.6
        
        # get pose
        transform = np.eye(4)

        # read points with down-sampling
        depth_limit = 6
        points_mat = back_project(depth, intrinsics, depth_limit=depth_limit, return_matrix=True)
        valid_map = [points_mat[:,:,2] > 0]
        points = points_mat[valid_map] # [N,3]
        points_rgb = image[valid_map]
        points_seg = seg_img_b[valid_map] # show segmentation

        # xixixi --- 0804
        '''
            to simulate the extreme case that i2p correspondences are poor
        '''
        points_rgb = points_rgb*0.2 #*0.2 #np.zeros_like(points_rgb)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(points_seg[:,:3])
        pcd.normals = o3d.utility.Vector3dVector(points_rgb[:,:3])
        # pcd = pcd.voxel_down_sample(voxel_size=0.015)
        # pcd = pcd.uniform_down_sample(8) # if use voxel down sample, unique has error?
        pcd = pcd.uniform_down_sample(2)
        points = np.array(pcd.points)
        points_rgb = np.array(pcd.normals)
        points_seg = np.array(pcd.colors)

        '''
            for visulization in paper writing
        '''
        is_need_vis_a = False
        # is_need_vis_a = True
        if is_need_vis_a:
            cv2.imshow("image", image)
            cv2.imshow("seg_img_a", seg_img_a)
            cv2.waitKey(0)
            pcda = o3d.geometry.PointCloud()
            pcda.points = o3d.utility.Vector3dVector(points[:,:3])
            pcda.colors = o3d.utility.Vector3dVector(points_rgb[:,:3])
            # pcda.colors = o3d.utility.Vector3dVector(points_seg[:,:3])
            show_pcd(pcda)
            print("points size: ", points.shape)
            assert 1==-1

        sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
        if self.max_points is not None and points.shape[0] > self.max_points:
            points = points[sel_indices]
            points_rgb = points_rgb[sel_indices]
            points_seg = points_seg[sel_indices]

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
            '''
                meanwhile revise bear image & segmentation image
            '''
            seg_img_a = cv2.resize(seg_img_a, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            seg_img_a = seg_img_a[start_h:end_h, start_w:end_w]

        '''
            0524: build topological relations gt labels
            segment 2d/3d
                --- idx (id)
                --- color code (cc)
                --- mask 
                --- center (ct)
            topological relations gt labels
                --- idx numpy array n*2 
        '''
        seg_2d_id, seg_2d_cc, seg_2d_mask, seg_2d_ct = \
            self.analyze_sam_2d(seg_img_a)
        seg_3d_id, seg_3d_cc, seg_3d_mask, seg_3d_ct = \
            self.analyze_sam_3d(points, points_seg)
        align_list, mat_primitive = \
            self.align_sam_2d_3d(seg_2d_id, seg_3d_id, seg_2d_cc, seg_3d_cc)
        # build data dict for topological relations
        data_dict["seg_2d_id"]   = np.array(seg_2d_id).astype(np.float32)
        data_dict["seg_2d_cc"]   = np.array(seg_2d_cc).astype(np.float32)
        data_dict["seg_2d_mask"] = np.array(seg_2d_mask).astype(np.float32)
        data_dict["seg_2d_ct"]   = np.array(seg_2d_ct).astype(np.float32)

        data_dict["seg_3d_id"]   = np.array(seg_3d_id).astype(np.float32)
        data_dict["seg_3d_cc"]   = np.array(seg_3d_cc).astype(np.float32)
        data_dict["seg_3d_mask"] = np.array(seg_3d_mask).astype(np.float32)
        data_dict["seg_3d_ct"]   = np.array(seg_3d_ct).astype(np.float32)

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
                check the gt correspondences in tum dataset --- ok
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
        '''
            add predicted depthes for i2p-pruning 
        '''
        if is_need_depth_v2:
            data_dict["depth_pred"] = depth_pred.astype(np.float32)
            data_dict["ratio_mat"]  = ratio_mat.astype(np.float32)
        '''
            add new features for the usage of segmentation 
        '''
        data_dict["seg_img_a"] = seg_img_a.astype(np.float32)
        data_dict["points_seg"] = points_seg.astype(np.float32)
        data_dict["pixel_coords"] = pixel_coords.astype(np.float32)
        '''
            add primitive matching matrix for supervision
        '''
        data_dict["mat_primitive_gt"] = mat_primitive.astype(np.float32)
        return data_dict

