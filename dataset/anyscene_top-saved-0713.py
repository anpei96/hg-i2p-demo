import os.path as osp
import random
from typing import Optional

import cv2
import time
import copy
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
    back_project)
from vision3d.utils.io import load_pickle, read_depth_image, read_image
from .utils_top_alignment import top_align_solver

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
        # scene_spec_name = "chess"       # using sampo instead of sampd
        # scene_spec_name = "office"
        scene_spec_name = "redkitchen"    # using sampo instead of sampd

        if is_use_scene_train:
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] == scene_spec_name]
        else:
            self.metadata_list = [x for x in self.metadata_list if x["scene_name"] != scene_spec_name]
            
        '''
            generate pixel arrays 
        '''
        metadata: dict = self.metadata_list[0]
        intrinsics_file = osp.join(self.data_dir, metadata["scene_name"], "camera-intrinsics.txt")
        intrinsics = np.loadtxt(intrinsics_file)
        fx = intrinsics[0,0]
        fy = intrinsics[1,1]
        cx = intrinsics[0,2]
        cy = intrinsics[1,2]
        image = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=True)
        self.pixel_coords = np.zeros((image.shape[0], image.shape[1], 2))
        for u in range(image.shape[1]):
            for v in range(image.shape[0]):
                self.pixel_coords[v,u,0] = u #(u-cx)/fx
                self.pixel_coords[v,u,1] = v #(v-cy)/fy
        
        '''
            introduce 2d-3d topological relationship alignment
        '''
        self.top_align_solver = top_align_solver(intrinsics)

    def __len__(self):
        return len(self.metadata_list)

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

    def __getitem__(self, index: int):
        data_dict = {}

        metadata: dict = self.metadata_list[index]
        data_dict["scene_name"] = metadata["scene_name"]
        data_dict["image_file"] = metadata["image_file"]
        data_dict["depth_file"] = metadata["depth_file"]
        data_dict["image_id"] = index #_get_frame_name(metadata["image_file"])
        data_dict["cloud_id"] = index #_get_frame_name(metadata["image_file"])

        intrinsics_file = osp.join(self.data_dir, metadata["scene_name"], "camera-intrinsics.txt")
        intrinsics = np.loadtxt(intrinsics_file)

        # read image
        pixel_coords = self.pixel_coords # [H,W,3]
        depth = read_depth_image(osp.join(self.data_dir, metadata["depth_file"])).astype(np.float)
        image = read_image(osp.join(self.data_dir, metadata["image_file"]), as_gray=False)
        data_dict["image_h"] = image.shape[0]
        data_dict["image_w"] = image.shape[1]

        '''
            depth correlation especially in self-collected
        '''
        # d_max = np.max(depth)
        # depth = depth * 6000.0/d_max

        '''
            anpei add load segment anything model
        '''
        _path = str(osp.join(self.data_dir, metadata["image_file"]))
        # seg_path_a = _path[:-4] + "_sampo.png" # hard-level test
        seg_path_a = _path[:-4] + "_sampo.png" # easy-level test o d
        seg_path_b = _path[:-4] + "_sampo.png"
        seg_img_a  = read_image(seg_path_a, as_gray=False) # [0,1]
        seg_img_b  = read_image(seg_path_b, as_gray=False) # [0,1]

        # read points with down-sampling
        '''
            directly using point cloud back-projected from depth image xyz+rgb
        '''
        depth_limit = 6.0
        points_mat = back_project(depth, intrinsics, depth_limit=depth_limit, return_matrix=True)
        valid_map = [points_mat[:,:,2] > 0]
        points = points_mat[valid_map]
        points_rgb = image[valid_map]     # show rgb color
        points_seg = seg_img_b[valid_map] # show segmentation

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(points_seg[:,:3])
        pcd.normals = o3d.utility.Vector3dVector(points_rgb[:,:3])
        # pcd = pcd.voxel_down_sample(voxel_size=0.015)
        pcd = pcd.uniform_down_sample(8) # if use voxel down sample, unique has error?
        # pcd = pcd.uniform_down_sample(2)
        points = np.array(pcd.points)
        points_rgb = np.array(pcd.normals)
        points_seg = np.array(pcd.colors)
        # print("===> points: ", points.shape)

        '''
            note-0521 add a new module of 2d-3d alignment for topology relations
                although it works sometime, but it is too technique sound, and 
                does not have the sufficient theoretical contribution

            therefore, it is recommanded to learn topology relations
        '''
        # ta = time.time()
        # self.top_align_solver.align(seg_img_a, points, points_seg)
        # tb = time.time()
        # print("==> topology align time: ", tb-ta)
        # assert 1==-1

        '''
            for visulization in paper writing
        '''
        is_need_vis_a = False
        # is_need_vis_a = True
        if is_need_vis_a:
            _path = str(osp.join(self.data_dir, metadata["image_file"]))
            image_rgb = cv2.imread(_path)
            cv2.imshow("rgb", image_rgb)
            cv2.imshow("seg", seg_img_a)
            cv2.waitKey(0)
            # pcda = o3d.geometry.PointCloud()
            # pcda.points = o3d.utility.Vector3dVector(points[:,:3])
            # pcda.colors = o3d.utility.Vector3dVector(points_rgb[:,:3])
            # pcda.colors = o3d.utility.Vector3dVector(points_seg[:,:3])
            # show_pcd(pcda)
            # print("points size: ", points.shape)
            # assert 1==-1

        sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
        if self.max_points is not None and points.shape[0] > self.max_points:
            points = points[sel_indices]
            points_rgb = points_rgb[sel_indices]
            points_seg = points_seg[sel_indices]

        '''
            transformation correction as identity matrix for using rgb-d pair
        '''
        transform = np.eye(4)

        '''
            visulization of colorful point cloud
        '''
        # a, b = np.unique(points_seg, return_inverse=True)
        # print("a, b: ", a.shape, b.shape)
        # c, d = np.unique(seg_img_a, return_inverse=True)
        # print("c, d: ", c.shape, d.shape)
        # print("points: ", points.shape)
        # print("transform: ", transform)
        # show_pcd(pcd)
        # assert 1==-1

        if self.use_augmentation:
            # augment point cloud
            '''
                it does not change the segmentation labels
            '''
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

