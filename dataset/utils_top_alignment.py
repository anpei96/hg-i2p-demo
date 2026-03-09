'''
module:   top_align
function: a python script to align 2d-3d topology relations without deep learning
author:   anpei
email:    anpei96@hust.edu.cn
date:     05.21.2025
'''

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
    back_project,
    render)
from vision3d.utils.io import load_pickle, read_depth_image, read_image
from vision3d.array_ops import axis_angle_to_rotation_matrix, get_transform_from_rotation_translation

def visualize_depth_map(dep):
    dep_min = np.min(dep)
    dep_max = np.max(dep)
    dep = (dep-dep_min)/(dep_max-dep_min)*255
    dep = dep.astype(np.uint8)
    dep_vis = cv2.applyColorMap(dep, cv2.COLORMAP_JET)
    return dep_vis

class top_align_solver:
    def __init__(self, intrinsics):
        self.intrinsics = intrinsics

    def align_error_eva(self):
        pass

    def align(self, seg_img, pts, seg_pts):
        '''
            seg_img:  (480, 640, 3)
            pts:      (35668, 3)
            seg_pts:  (35668, 3)
        '''
        pixel_coords = np.zeros((seg_img.shape[0], seg_img.shape[1], 2))
        for u in range(seg_img.shape[1]):
            for v in range(seg_img.shape[0]):
                pixel_coords[v,u,0] = u 
                pixel_coords[v,u,1] = v 

        '''
            step one: extract each sub-seg 2d/3d centers
                does not consider the black region
        '''
        # image processing branch
        sa = (seg_img * 255).astype(np.uint8) # [H,W,3]
        sa_v = sa[:,:,0]*1e6 + sa[:,:,1]*1e3 + sa[:,:,2]
        i_a, i_b = np.unique(sa_v, return_inverse=True)  
        num_img_seg = len(i_a)

        seg_center_2d, seg_num_2d = [], []
        seg_mask_2d = []
        seg_num_max_2d = 0
        for i in range(num_img_seg):
            if i_a[i] < 1.0:
                continue # filter the un-segment region
            mask_i = (i_b == i).reshape(seg_img.shape[0], seg_img.shape[1])
            coords = pixel_coords[mask_i]
            mean_coords = np.median(coords, axis=0) # np.mean median

            seg_center_2d.append(mean_coords)
            seg_num_2d.append(coords.shape[0])
            seg_mask_2d.append(mask_i)

            if coords.shape[0] > seg_num_max_2d:
                seg_num_max_2d = coords.shape[0]
            # print("coords: ", coords.shape)
            # print("mean_coords: ", mean_coords.shape)
            # print("num_pixels: ", coords.shape[0])
            # print("seg_num_max_2d: ", seg_num_max_2d)

        # point cloud processing branch
        sb = (seg_pts * 255).astype(np.uint8) # [N,3]
        sb_v = sb[:,0]*1e6 + sb[:,1]*1e3 + sb[:,2]
        p_a, p_b  = np.unique(sb_v, return_inverse=True)
        num_pcd_seg = len(p_a)

        seg_center_3d, seg_num_3d = [], []
        seg_mask_3d = []
        seg_num_max_3d = 0
        for i in range(num_pcd_seg):
            if p_a[i] < 1.0:
                continue # filter the un-segment region
            mask_i = (p_b == i)
            coords = pts[mask_i]
            mean_coords = np.median(coords, axis=0) # np.mean median
            seg_center_3d.append(mean_coords)
            seg_num_3d.append(coords.shape[0])
            seg_mask_3d.append(mask_i)

            if coords.shape[0] > seg_num_max_3d:
                seg_num_max_3d = coords.shape[0]
            # print("coords: ", coords.shape)
            # print("mean_coords: ", mean_coords.shape)
            # print("num_pixels: ", coords.shape[0])
            # print("seg_num_max_3d: ", seg_num_max_3d)
        
        '''
            step two: corase align each sub-seg 2d/3d centers --- test ok
        '''
        center_2d = np.array(seg_center_2d) # [n,2]
        center_3d = np.array(seg_center_3d) # [n,3]
        scores_2d = np.array(seg_num_2d)/seg_num_max_2d
        scores_3d = np.array(seg_num_3d)/seg_num_max_3d

        num_img_seg = center_2d.shape[0]
        num_pcd_seg = center_3d.shape[0]
        ini_cor_mat = np.ones((num_img_seg, num_pcd_seg))*(-1)
        ca_center_3d = []
        ca_scores_3d = []
        for i in range(num_img_seg):
            best_i_can = 0
            best_i_sc  = 99
            for j in range(num_pcd_seg):
                ini_cor_mat[i,j] = np.abs(scores_2d[i] - scores_3d[j])
                if ini_cor_mat[i,j] <= best_i_sc:
                    best_i_can = j
                    best_i_sc = ini_cor_mat[i,j]
            ca_center_3d.append(center_3d[best_i_can,:])
            ca_scores_3d.append(best_i_sc)

        ca_center_3d = np.array(ca_center_3d) # [n,3]
        
        success, rvec, tvec, inliers = \
            cv2.solvePnPRansac(ca_center_3d, center_2d, self.intrinsics, None,
            iterationsCount=5000, reprojectionError=10, flags=cv2.SOLVEPNP_P3P)

        # inliers check
        # opt_pts_3d = []
        # opt_pix_2d = []
        # for i in range(len(inliers)):
        #     index = inliers[i]
        #     if ca_scores_3d[i] <= 0.05:
        #         opt_pts_3d.append(ca_center_3d[index, :])
        #         opt_pix_2d.append(center_2d[index, :])
        # opt_pix_2d = np.array(opt_pix_2d) # [n,2]
        # opt_pts_3d = np.array(opt_pts_3d) # [n,3]

        # success, rvec, tvec, inliers = \
        #     cv2.solvePnPRansac(opt_pts_3d, opt_pix_2d, self.intrinsics, None,
        #     iterationsCount=10000, 
        #     useExtrinsicGuess=True, reprojectionError=10, rvec=rvec, tvec=tvec)
        
        rvec = rvec[:, 0]
        tvec = tvec[:, 0]
        rotation = axis_angle_to_rotation_matrix(rvec)
        transform = get_transform_from_rotation_translation(rotation, tvec)
        # print("rvec: ", rvec)
        # print("tvec: ", tvec)
        # print("success: ", success)
        # print("inliers: ", len(inliers))
        # print("transform")
        # print(transform)

        '''
            step three: accurate align each sub-seg 2d/3d centers --- test
                using seg_mask_2d and seg_mask_3d
        '''
        # transform = np.eye(4)
        # transform and estimated_transform
        proj_center_3d = render(center_3d, self.intrinsics, transform) # [n,2]
        tmp = proj_center_3d[:,0]*1 
        proj_center_3d[:,0] = proj_center_3d[:,1]
        proj_center_3d[:,1] = tmp
        acc_num = 0
        all_num = num_img_seg

        # generate gt mask-3d
        mask_3d_gt = []
        for j in range(num_pcd_seg):
            mask_3d_j = np.zeros_like(seg_mask_2d[0])
            # print("mask_3d_j: ", mask_3d_j.shape)
            pts_3d_j  = pts[seg_mask_3d[j]]
            proj_pts_3d_j = render(pts_3d_j, self.intrinsics, np.eye(4))
            proj_pts_3d_j = proj_pts_3d_j.astype(np.int)
            for k in range(proj_pts_3d_j.shape[0]):
                u = proj_pts_3d_j[k,0]
                v = proj_pts_3d_j[k,1]
                if ((u<0) | (u>=mask_3d_j.shape[0])):
                    continue
                if ((v<0) | (v>=mask_3d_j.shape[1])):
                    continue
                mask_3d_j[u,v] = 1
            mask_3d_gt.append(mask_3d_j)

        # pre-compute each mask-3d
        mask_3d = []
        for j in range(num_pcd_seg):
            mask_3d_j = np.zeros_like(seg_mask_2d[0])
            # print("mask_3d_j: ", mask_3d_j.shape)
            pts_3d_j  = pts[seg_mask_3d[j]]
            proj_pts_3d_j = render(pts_3d_j, self.intrinsics, transform)
            proj_pts_3d_j = proj_pts_3d_j.astype(np.int)
            for k in range(proj_pts_3d_j.shape[0]):
                u = proj_pts_3d_j[k,0]
                v = proj_pts_3d_j[k,1]
                if ((u<0) | (u>=mask_3d_j.shape[0])):
                    continue
                if ((v<0) | (v>=mask_3d_j.shape[1])):
                    continue
                mask_3d_j[u,v] = 1
            mask_3d.append(mask_3d_j)

        for i in range(num_img_seg):
            best_i_can = 0
            best_i_iou = -1
            mask_2d_i  = seg_mask_2d[i]
            for j in range(num_pcd_seg):
                mask_3d_j = mask_3d[j]
                iou = np.sum(mask_2d_i & mask_3d_j)\
                    /np.sum(mask_2d_i | mask_3d_j)
                if iou >= best_i_iou:
                    best_i_can = j
                    best_i_iou = iou

            # print("best match: ", i, " --- ", best_i_can, " iou: ", best_i_iou)
            # generate gt match
            best_i_can_gt = 0
            best_i_iou_gt = -1
            for j in range(num_pcd_seg):
                mask_3d_j = mask_3d_gt[j]
                iou = np.sum(mask_2d_i & mask_3d_j)\
                    /np.sum(mask_2d_i | mask_3d_j)
                if iou >= best_i_iou_gt:
                    best_i_can_gt = j
                    best_i_iou_gt = iou
            
            if best_i_can == best_i_can_gt:
                acc_num = acc_num+1
        
        print("acc: ", acc_num/num_img_seg, " acc_num: ", acc_num, " num_img_seg: ", num_img_seg)

        # debug
        img_render = seg_img-seg_img
        for i in range(num_pcd_seg):
            u = int(proj_center_3d[i,0])
            v = int(proj_center_3d[i,1])
            cv2.circle(img_render, (int(u), int(v)), 6, (0,0,255), -1)
        for i in range(num_img_seg):
            u = int(center_2d[i,0])
            v = int(center_2d[i,1])
            cv2.circle(img_render, (int(u), int(v)), 3, (0,255,0), -1)

        res_img = cv2.addWeighted(seg_img, 0.5, img_render, 0.5, 0.5)

        # print("rep error: ", proj_center_3d - center_2d)
        # print(ini_cor_mat)
        # cv2.imshow("img_render", img_render)
        # cv2.imshow("seg_img", seg_img)
        # cv2.imshow("res_img", res_img)
        # cv2.waitKey()
        # assert 1==-1