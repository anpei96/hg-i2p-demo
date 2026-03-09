import os.path as osp
import random
from typing import Optional

import glob
import copy
import cv2
import open3d as o3d
import numpy as np
from torch.utils.data import Dataset

import trimesh
import csv
import glob
import os
from PIL import Image

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

def show_pcd(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window("point cloud")
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    render_options.background_color = np.array([0,0,0])
    render_options.point_size = 1.0
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run() 

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
                "transform_matrix": (np.linalg.inv(T)),
            }

            self.frames.append(frame)

if __name__ == '__main__':
    '''
        a simple pipeline to reconstruct a whole pcd file
    '''
    data_dir = "/media/anpei/DiskA/05_i2p_fewshot/data/Tum/scene_01/"
    # data_dir = "/media/anpei/DiskA/05_i2p_fewshot/data/Tum/scene_02/"
    # data_dir = "/media/anpei/DiskA/05_i2p_fewshot/data/Tum/scene_03/"
    tumparser = TUMParser(data_dir)

    # read intrinsic matrix
    intrinsics = np.zeros((3,3))
    intrinsics[2,2] = 1.0
    # ==> scene_01
    intrinsics[0,0] = 517.306408
    intrinsics[1,1] = 516.469215
    intrinsics[0,2] = 318.643040
    intrinsics[1,2] = 255.313989
    # ==> scene_02
    # intrinsics[0,0] = 520.90862
    # intrinsics[1,1] = 521.007327
    # intrinsics[0,2] = 325.141442
    # intrinsics[1,2] = 249.701764
    # # ==> scene_03
    # intrinsics[0,0] = 535.4
    # intrinsics[1,1] = 539.2
    # intrinsics[0,2] = 320.1
    # intrinsics[1,2] = 247.6

    # create an empty point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.zeros((0,3)))
    pcd.colors = o3d.utility.Vector3dVector(np.zeros((0,3)))

    num_sample = tumparser.n_img
    for data_index in range(num_sample):
        '''
            note-0628 gt pose error of tum dataset
                scene_01 max frames 120  interval 10
                scene_02 max frames 3600 interval 100
                scene_03 max frames 2500 interval 50
        '''
        # for debug
        if data_index == 250:
            break
        # sample with interval
        # if data_index <= 50:
        #     continue
        if data_index % 20 !=0:
            continue

        print("=> processing: ", data_index)
        data_info = tumparser.frames[data_index]
        image_path = data_info["file_path"]
        image = cv2.imread(image_path)
        image_bgr  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/ 255.0
        depth_path = data_info["depth_path"]
        depth = read_depth_image(depth_path, scaling_factor=5).astype(np.float)

        tf = data_info["transform_matrix"]
        depth_limit = 6
        points_mat = back_project(depth, intrinsics, depth_limit=depth_limit, return_matrix=True)
        valid_map = [points_mat[:,:,2] > 0]
        points = points_mat[valid_map] # [N,3]
        points_rgb = image_bgr[valid_map]

        # coordinate transformation
        tf_inv = np.linalg.inv(tf)
        # print(tf_inv)
        pts  = points.T    # [3,N]
        rmat = tf_inv[0:3,0:3] # [3,3]
        tvec = tf_inv[0:3,3:4] # [3,1]
        pts  = np.matmul(rmat, pts) 
        pts  = pts + tvec
        points_tf = pts.T  # [N,3]

        # pcd add
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points_tf)
        pcd_temp.colors = o3d.utility.Vector3dVector(points_rgb)
        pcd = pcd + pcd_temp
        # pcd = pcd.voxel_down_sample(voxel_size=0.015)
    
    # pcd = pcd.voxel_down_sample(voxel_size=0.005)
    show_pcd(pcd)

    # save pcd
    # o3d.io.write_point_cloud(data_dir+"map.pcd", pcd)
    # print("save map.pcd in file: ", data_dir+"map.pcd")
