import os.path as osp
import random
from typing import Optional

import cv2
import time
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
        # scene_spec_name = "chess"
        scene_spec_name = "office"
        # scene_spec_name = "redkitchen"

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

    def __len__(self):
        return len(self.metadata_list)

    def mask_generate_from_seg_single(self, pts_seg):
        sb = (pts_seg * 255).astype(np.uint8) # [N,3]
        sb_v = sb[:,0]*1e6 + sb[:,1]*1e3 + sb[:,2]
        p_a, p_b  = np.unique(sb_v, return_inverse=True)
        num_pts_seg = len(p_a)
        pts_seg_mask = []
        for j in range(num_pts_seg):
            mask_j = (p_b == j)
            pts_seg_mask.append(mask_j)
        return num_pts_seg, pts_seg_mask, p_a

    def mask_generate_from_seg_single_img(self, seg_img_a):
        sa = (seg_img_a * 255).astype(np.uint8) # [H,W,3]
        sa_v = sa[:,:,0]*1e6 + sa[:,:,1]*1e3 + sa[:,:,2]
        i_a, i_b = np.unique(sa_v, return_inverse=True)
        num_pts_seg = len(i_a)
        img_seg_mask = []
        is_detect_black = False
        for i in range(num_pts_seg):
            if i == 0:
                if i_a[i] < 1.0:
                    is_detect_black = True
                    continue # detect black region
            mask_i = (i_b == i)
            img_seg_mask.append(mask_i)
        if is_detect_black == True:
            num_img_seg = num_img_seg - 1
        return num_pts_seg, img_seg_mask, i_a

    def mask_generate_from_seg(self, seg_img_a, seg_img_b):
        sa = (seg_img_a * 255).astype(np.uint8) # [H,W,3]
        sb = (seg_img_b * 255).astype(np.uint8) # [H,W,3]
        sa_v = sa[:,:,0]*1e6 + sa[:,:,1]*1e3 + sa[:,:,2]
        sb_v = sb[:,:,0]*1e6 + sb[:,:,1]*1e3 + sb[:,:,2]
        i_a, i_b = np.unique(sa_v, return_inverse=True)
        p_a, p_b = np.unique(sb_v, return_inverse=True)
        num_img_seg, num_pts_seg = len(i_a), len(p_a)
        img_seg_mask = []
        pts_seg_mask = []
        is_detect_black = False
        for i in range(num_img_seg):
            '''
                in image segmentation, due to real-time requirment,
                    there are some region without any segmenation and mark as black
                    we attempt to filter these black region
            '''
            if i == 0:
                if i_a[i] < 1.0:
                    is_detect_black = True
                    # print("debug: ", np.sum((i_b == i)))
                    continue # detect black region
            mask_i = (i_b == i)
            img_seg_mask.append(mask_i)
        for j in range(num_pts_seg):
            mask_j = (p_b == j)
            pts_seg_mask.append(mask_j)
        if is_detect_black == True:
            num_img_seg = num_img_seg - 1
        return num_img_seg, num_pts_seg, img_seg_mask, pts_seg_mask, p_a, i_a

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
            anpei add load segment anything model
        '''
        _path = str(osp.join(self.data_dir, metadata["image_file"]))
        # seg_path_a = _path[:-4] + "_sampo.png" # hard-level test
        seg_path_a = _path[:-4] + "_sampd.png" # easy-level test
        seg_path_b = _path[:-4] + "_sampd.png"
        seg_img_a  = read_image(seg_path_a, as_gray=False) # [0,1]
        seg_img_b  = read_image(seg_path_b, as_gray=False) # [0,1]
        '''
            generate segmentation matching matrix labels --- step one --- ok
        '''
        num_img_seg, num_pts_seg, img_seg_mask, pts_seg_mask, p_old, q_old = \
            self.mask_generate_from_seg(seg_img_a, seg_img_b)
        mat_primitive = np.zeros((num_img_seg, num_pts_seg))
        for i in range(num_img_seg):
            for j in range(num_pts_seg):
                mask_img, mask_pts = img_seg_mask[i], pts_seg_mask[j]
                cnt_a = np.sum(mask_img & mask_pts)
                a, b = np.sum(mask_img), np.sum(mask_pts)
                cnt_b = np.max((a,b))
                iou = cnt_a/cnt_b
                if iou > 0.75: mat_primitive[i,j] = 1
                # print(i, " --- ", j, " : ", iou)
        # print("mat_primitive: ", mat_primitive.shape, np.sum(mat_primitive))

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
        points = np.array(pcd.points)
        points_rgb = np.array(pcd.normals)
        points_seg = np.array(pcd.colors)
        # print("===> points: ", points.shape)

        '''
            for visulization in paper writing
        '''
        is_need_vis_a = False
        # is_need_vis_a = True
        if is_need_vis_a:
            _path = str(osp.join(self.data_dir, metadata["image_file"]))
            image_rgb = cv2.imread(_path)
            # cv2.imshow("rgb", image_rgb)
            # cv2.imshow("seg", seg_img_a)
            # cv2.waitKey(0)
            # pcda = o3d.geometry.PointCloud()
            # pcda.points = o3d.utility.Vector3dVector(points[:,:3])
            # pcda.colors = o3d.utility.Vector3dVector(points_rgb[:,:3])
            # show_pcd(pcda)
            assert 1==-1

        sel_indices = np.random.permutation(points.shape[0])[: self.max_points]
        if self.max_points is not None and points.shape[0] > self.max_points:
            points = points[sel_indices]
            points_rgb = points_rgb[sel_indices]
            points_seg = points_seg[sel_indices]
        '''
            generate segmentation matching matrix labels --- step two --- ok
        '''
        num_pts_seg_a, pts_seg_mask_a, p_new = self.mask_generate_from_seg_single(points_seg)
        if num_pts_seg_a < num_pts_seg:
            '''
                a extreme case, we need to reduce some columns of mat_primitive
            '''
            remove_id_old = []
            for id_old in range(num_pts_seg):
                is_find = False
                for id_new in range(num_pts_seg_a):
                    color_old = p_old[id_old]
                    color_new = p_new[id_new]
                    if color_new == color_old:
                        is_find = True
                if is_find == False:
                    remove_id_old.append(id_old)
            mat_primitive = np.delete(mat_primitive, obj=remove_id_old, axis=1)

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
            pixel_coords = cv2.resize(pixel_coords, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            pixel_coords = pixel_coords[start_h:end_h, start_w:end_w]
            seg_img_a = cv2.resize(seg_img_a, (new_image_w, new_image_h), interpolation=cv2.INTER_LINEAR)
            seg_img_a = seg_img_a[start_h:end_h, start_w:end_w]
            '''
                generate segmentation matching matrix labels --- step three --- ok
            '''
            num_img_seg_b, img_seg_mask_b, q_new = self.mask_generate_from_seg_single_img(seg_img_a)
            if num_img_seg_b < num_img_seg:
                '''
                    a extreme case, we need to reduce some rows of mat_primitive
                '''
                remove_id_old = []
                for id_old in range(num_img_seg):
                    is_find = False
                    for id_new in range(num_img_seg_b):
                        color_old = q_old[id_old]
                        color_new = q_new[id_new]
                        if color_new == color_old:
                            is_find = True
                    if is_find == False:
                        remove_id_old.append(id_old)
                mat_primitive = np.delete(mat_primitive, obj=remove_id_old, axis=0)

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

