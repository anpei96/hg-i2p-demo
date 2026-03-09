'''
    project: an unified solution to cross-modality visual odometry (c-vo)
    author:  anpei 
    email:   anpei96@hust.edu.cn
'''

import glob
import json
import os.path as osp
import time
import numpy as np
import torch
import cv2 as cv
from vision3d.ops import apply_transform
from vision3d.utils.tensor import tensor_to_array
from vision3d.utils.opencv import registration_with_pnp_ransac

def correspondence_visulize(data_dict, base_path, output_dict, result_dict):
    image_file = data_dict["image_file"]
    transform  = data_dict["transform"]
    intrinsics = data_dict["intrinsics"]
    pts        = data_dict["points"][0]
    pts_rgb    = data_dict["points_rgb"]

    '''
        1. visulize rgb/depth image
    '''
    print("===> : ", base_path + image_file)
    rgb_image = cv.imread(base_path + image_file)

    img_h = rgb_image.shape[0]
    img_w = rgb_image.shape[1]

    pts = apply_transform(pts, transform)
    pix = (torch.matmul(intrinsics, pts.T)).T
    pix = pix.cpu().numpy()
    dep = pix[:,2:3]
    pix = pix/dep
    pix = pix.astype(np.int)
    num = pix.shape[0]
    d_max, d_min = np.max(dep), np.min(dep)

    pts_image = np.zeros_like(rgb_image)
    for i in range(num):
        u = int(pix[i,0])
        v = int(pix[i,1])
        r = int(pts_rgb[i,0]*255)
        g = int(pts_rgb[i,1]*255)
        b = int(pts_rgb[i,2]*255)
        if ((u < 0) | (u >= img_w)):
            continue
        if ((v < 0) | (v >= img_h)):
            continue
        cv.circle(pts_image, (int(u), int(v)), 3, (b,g,r), -1)
    
    vis_rgb_pts_img = np.concatenate((rgb_image, pts_image), axis=1)
    vis_mix_img = cv.addWeighted(rgb_image, 0.5, pts_image, 0.5, 0)
    vis_mix_img = pts_image

    '''
        2. visulize 2d/3d corner points
    '''
    img_corr_pixels=tensor_to_array(output_dict["img_corr_pixels"])
    pcd_corr_pixels=tensor_to_array(output_dict["pcd_corr_pixels"])
    corr_scores=tensor_to_array(output_dict["corr_scores"])
    num_pts = img_corr_pixels.shape[0]
    for i in range(num_pts):
        # if corr_scores[i] >= 0:
        #     continue
        u = int(img_corr_pixels[i,0])
        v = int(img_corr_pixels[i,1])
        cv.circle(vis_rgb_pts_img, (int(v), int(u)), 1, (0,255,255), -1)
    
    pts_corr = apply_transform(
        torch.tensor(pcd_corr_pixels).cuda().float(), transform)
    pix_corr = (torch.matmul(intrinsics, pts_corr.T)).T
    pix_corr = pix_corr.cpu().numpy()
    dep_corr = pix_corr[:,2:3]
    pix_corr = pix_corr/dep_corr
    pix_corr = pix_corr.astype(np.int)
    for i in range(num_pts):
        # if corr_scores[i] >= 0:
        #     continue
        u = int(pix_corr[i,0])
        v = int(pix_corr[i,1])
        cv.circle(vis_rgb_pts_img, (int(u)+img_w, int(v)), 1, (0,255,255), -1)

    '''
        3. visulize 2d/3d point correspondence
    '''     
    for i in range(num_pts):
        # if corr_scores[i] >= 0:
        #     continue
        u_img = int(img_corr_pixels[i,1])
        v_img = int(img_corr_pixels[i,0])
        u_pts = int(pix_corr[i,0])
        v_pts = int(pix_corr[i,1])
        d = np.abs(u_img - u_pts) + np.abs(v_img - v_pts)
        th = 15
        if d > th:
            cv.line(vis_rgb_pts_img, 
                (u_img, v_img), (u_pts+img_w, v_pts), (0,0,255), 1)
        if d <= th:
            cv.line(vis_rgb_pts_img, 
                (u_img, v_img), (u_pts+img_w, v_pts), (0,255,0), 1)
    
    '''
        4. visulize PIR/IR in the image
    ''' 
    # res_string = "IR: "+ str(result_dict['IR'].cpu().numpy())
    res_string = "IR: "+ format(result_dict['IR'].cpu().numpy(), '.3f')
    cv.putText(vis_rgb_pts_img, res_string, 
        (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    # print("img_corr_pixels: ", img_corr_pixels.shape)
    # print("pcd_corr_pixels: ", pcd_corr_pixels.shape)
    # print("result_dict")
    # print(result_dict)

    cv.imshow("vis_rgb_pts_img", vis_rgb_pts_img)
    cv.imshow("vis_mix_img", vis_mix_img)
    cv.waitKey(0)
    # assert 1==-1
    return vis_rgb_pts_img

def data_dict_copy_npz(data_dict):
    data_dict_eva = {}
    data_dict_eva["image_file"] = data_dict["image_file"]
    data_dict_eva["depth_file"] = data_dict["depth_file"]
    data_dict_eva["pcd_points"] = (data_dict["pcd_points"])
    data_dict_eva["pcd_points_f"] = (data_dict["pcd_points_f"])
    data_dict_eva["pcd_points_c"] = (data_dict["pcd_points_c"])
    data_dict_eva["img_num_nodes"] = data_dict["img_num_nodes"]
    data_dict_eva["pcd_num_nodes"] = data_dict["pcd_num_nodes"]

    data_dict_eva["img_node_corr_indices"] = (data_dict["img_node_corr_indices"])
    data_dict_eva["pcd_node_corr_indices"] = (data_dict["pcd_node_corr_indices"])
    data_dict_eva["img_node_corr_levels"]  = (data_dict["img_node_corr_levels"])
    data_dict_eva["img_corr_points"] = (data_dict["img_corr_points"])
    data_dict_eva["pcd_corr_points"] = (data_dict["pcd_corr_points"])
    data_dict_eva["img_corr_pixels"] = (data_dict["img_corr_pixels"])
    data_dict_eva["pcd_corr_pixels"] = (data_dict["pcd_corr_pixels"])
    data_dict_eva["corr_scores"] = (data_dict["corr_scores"])
    
    data_dict_eva["gt_img_node_corr_indices"] = (data_dict["gt_img_node_corr_indices"])
    data_dict_eva["gt_pcd_node_corr_indices"] = (data_dict["gt_pcd_node_corr_indices"])
    data_dict_eva["gt_img_node_corr_overlaps"] = (data_dict["gt_img_node_corr_overlaps"])
    data_dict_eva["gt_pcd_node_corr_overlaps"] = (data_dict["gt_pcd_node_corr_overlaps"])
    data_dict_eva["gt_node_corr_min_overlaps"] = (data_dict["gt_node_corr_min_overlaps"])
    data_dict_eva["gt_node_corr_max_overlaps"] = (data_dict["gt_node_corr_max_overlaps"])
    
    data_dict_eva["transform"]  = (data_dict["transform"])
    data_dict_eva["intrinsics"] = (data_dict["intrinsics"])
    return data_dict_eva

def data_dict_transfer(data_dict, output_dict):
    data_dict_eva = {}

    data_dict_eva["image_file"] = data_dict["image_file"]
    data_dict_eva["depth_file"] = data_dict["depth_file"]
    data_dict_eva["pcd_points"] = tensor_to_array(output_dict["pcd_points"])
    data_dict_eva["pcd_points_f"] = tensor_to_array(output_dict["pcd_points_f"])
    data_dict_eva["pcd_points_c"] = tensor_to_array(output_dict["pcd_points_c"])
    data_dict_eva["img_num_nodes"] = output_dict["img_num_nodes"]
    data_dict_eva["pcd_num_nodes"] = output_dict["pcd_num_nodes"]

    data_dict_eva["img_node_corr_indices"] = tensor_to_array(output_dict["img_node_corr_indices"])
    data_dict_eva["pcd_node_corr_indices"] = tensor_to_array(output_dict["pcd_node_corr_indices"])
    data_dict_eva["img_node_corr_levels"]  = tensor_to_array(output_dict["img_node_corr_levels"])
    data_dict_eva["img_corr_points"] = tensor_to_array(output_dict["img_corr_points"])
    data_dict_eva["pcd_corr_points"] = tensor_to_array(output_dict["pcd_corr_points"])
    data_dict_eva["img_corr_pixels"] = tensor_to_array(output_dict["img_corr_pixels"])
    data_dict_eva["pcd_corr_pixels"] = tensor_to_array(output_dict["pcd_corr_pixels"])
    data_dict_eva["corr_scores"] = tensor_to_array(output_dict["corr_scores"])
    
    data_dict_eva["gt_img_node_corr_indices"] = tensor_to_array(output_dict["gt_img_node_corr_indices"])
    data_dict_eva["gt_pcd_node_corr_indices"] = tensor_to_array(output_dict["gt_pcd_node_corr_indices"])
    data_dict_eva["gt_img_node_corr_overlaps"] = tensor_to_array(output_dict["gt_img_node_corr_overlaps"])
    data_dict_eva["gt_pcd_node_corr_overlaps"] = tensor_to_array(output_dict["gt_pcd_node_corr_overlaps"])
    data_dict_eva["gt_node_corr_min_overlaps"] = tensor_to_array(output_dict["gt_node_corr_min_overlaps"])
    data_dict_eva["gt_node_corr_max_overlaps"] = tensor_to_array(output_dict["gt_node_corr_max_overlaps"])
    
    data_dict_eva["transform"]  = tensor_to_array(data_dict["transform"])
    data_dict_eva["intrinsics"] = tensor_to_array(data_dict["intrinsics"])
    return data_dict_eva

from vision3d.array_ops import (
    evaluate_correspondences,
    evaluate_sparse_correspondences,
    isotropic_registration_error,
    registration_rmse,
)

def eval_per_image(cfg, data_dict):
    pcd_points = data_dict["pcd_points"]
    img_num_nodes = data_dict["img_num_nodes"]
    pcd_num_nodes = data_dict["pcd_num_nodes"]
    img_node_corr_indices = data_dict["img_node_corr_indices"]
    pcd_node_corr_indices = data_dict["pcd_node_corr_indices"]

    img_corr_points = data_dict["img_corr_points"]
    pcd_corr_points = data_dict["pcd_corr_points"]
    img_corr_pixels = data_dict["img_corr_pixels"]
    pcd_corr_pixels = data_dict["pcd_corr_pixels"]
    corr_scores = data_dict["corr_scores"]

    gt_img_node_corr_indices = data_dict["gt_img_node_corr_indices"]
    gt_pcd_node_corr_indices = data_dict["gt_pcd_node_corr_indices"]
    transform = data_dict["transform"]

    # if args.num_corr is not None and corr_scores.shape[0] > args.num_corr:
    #     num_corr = corr_scores.shape[0]
    #     sel_indices = np.argsort(-corr_scores)[:num_corr]
    #     img_corr_points = img_corr_points[sel_indices]
    #     pcd_corr_points = pcd_corr_points[sel_indices]
    #     img_corr_pixels = img_corr_pixels[sel_indices]
    #     pcd_corr_pixels = pcd_corr_pixels[sel_indices]
    #     corr_scores = corr_scores[sel_indices]

    num_correspondences = img_corr_points.shape[0]

    if num_correspondences > 0:
        fine_matching_result_dict = evaluate_correspondences(
            pcd_corr_points, img_corr_points, transform, positive_radius=cfg.eval.acceptance_radius
        )
    else:
        fine_matching_result_dict = {"inlier_ratio": 0.0, "overlap": 0.0, "distance": 0.0}

    inlier_ratio = fine_matching_result_dict["inlier_ratio"]
    overlap = fine_matching_result_dict["overlap"]

    if num_correspondences >= 4:
        intrinsics = data_dict["intrinsics"]
        estimated_transform = registration_with_pnp_ransac(
            pcd_corr_points,
            img_corr_pixels,
            intrinsics,
            num_iterations=cfg.ransac.num_iterations,
            distance_tolerance=cfg.ransac.distance_tolerance)

        rmse = registration_rmse(pcd_points, transform, estimated_transform)
        registration_recall = float(rmse < cfg.eval.rmse_threshold)
        rre, rte = isotropic_registration_error(transform, estimated_transform)
    else:
        estimated_transform = np.eye(4)
        registration_recall = 0.0

    return inlier_ratio, registration_recall, rmse, rre, rte, estimated_transform

from numpy import ndarray
from typing import Optional
from vision3d.array_ops import axis_angle_to_rotation_matrix, get_transform_from_rotation_translation

def registration_with_pnp_ransac_ours(
    corr_points: ndarray,
    corr_pixels: ndarray,
    intrinsics: ndarray,
    distortion: Optional[ndarray] = None,
    num_iterations: int = 5000,
    distance_tolerance: float = 8.0,
    transposed: bool = True,
) -> ndarray:
    """PnP-RANSAC registration with OpenCV.

    Note:
        1. cv2.solvePnPRansac() requires the pixels are in the order of (w, h).

    Args:
        corr_points (array): a float array of the 3D correspondence points in the shape of (N, 3).
        corr_pixels (array): an int array of the 2D correspondence pixels in the shape of (N, 2).
        intrinsics (array): a float array of the camera intrinsics in the shape of (3, 3).
        distortion (array, optional): a float array of the distortion parameter in the shape of (4, 1) or (12, 1).
        num_iterations (int): the number of ransac iterations.
        distance_tolerance (float): the distance tolerance for ransac.
        transposed (bool): if True, the pixel coordinates are in the order of (h, w) or (w, h) otherwise.

    Returns:
        A float array of the estimated transformation from 3D to 2D.
    """
    if corr_points.shape[0] < 4:
        # too few correspondences, return None
        return None

    if distortion is None:
        distortion = np.zeros(shape=(4, 1))
    if transposed:
        corr_pixels = np.stack([corr_pixels[..., 1], corr_pixels[..., 0]], axis=-1)  # (h, w) -> (w, h)

    corr_points = corr_points.astype(np.float)
    corr_pixels = corr_pixels.astype(np.float)
    intrinsics = intrinsics.astype(np.float)

    _, axis_angle, translation, inliers = cv.solvePnPRansac(
        corr_points,
        corr_pixels,
        intrinsics,
        distortion,
        iterationsCount=num_iterations,
        reprojectionError=distance_tolerance,
        flags=cv.SOLVEPNP_P3P,
    )

    '''
        a bundle adjustment --- not always good :(
    '''
    # inliers check
    # opt_pts_3d = []
    # opt_pix_2d = []
    # for i in range(len(inliers)):
    #     index = inliers[i]
    #     opt_pts_3d.append(corr_points[index, :])
    #     opt_pix_2d.append(corr_pixels[index, :])
    # opt_pix_2d = np.array(opt_pix_2d) # [n,2]
    # opt_pts_3d = np.array(opt_pts_3d) # [n,3]
    # success, axis_angle, translation, inliers = \
    #     cv.solvePnPRansac(opt_pts_3d, opt_pix_2d, intrinsics, distortion,
    #     iterationsCount=num_iterations, 
    #     useExtrinsicGuess=True, reprojectionError=distance_tolerance, 
    #     rvec=axis_angle, tvec=translation, flags=cv.SOLVEPNP_P3P)

    axis_angle = axis_angle[:, 0]
    translation = translation[:, 0]
    rotation = axis_angle_to_rotation_matrix(axis_angle)
    estimated_transform = get_transform_from_rotation_translation(rotation, translation)

    return estimated_transform

def eval_per_image_np(cfg, data_dict, planB=False):
    pcd_points = data_dict['pcd_points']
    img_num_nodes = data_dict["img_num_nodes"]
    pcd_num_nodes = data_dict["pcd_num_nodes"]
    img_node_corr_indices = data_dict["img_node_corr_indices"]
    pcd_node_corr_indices = data_dict["pcd_node_corr_indices"]

    img_corr_points = data_dict["img_corr_points"]
    pcd_corr_points = data_dict["pcd_corr_points"]
    img_corr_pixels = data_dict["img_corr_pixels"]
    pcd_corr_pixels = data_dict["pcd_corr_pixels"]
    corr_scores = data_dict["corr_scores"]

    gt_img_node_corr_indices = data_dict["gt_img_node_corr_indices"]
    gt_pcd_node_corr_indices = data_dict["gt_pcd_node_corr_indices"]
    transform = data_dict["transform"]

    # if args.num_corr is not None and corr_scores.shape[0] > args.num_corr:
    #     num_corr = corr_scores.shape[0]
    #     sel_indices = np.argsort(-corr_scores)[:num_corr]
    #     img_corr_points = img_corr_points[sel_indices]
    #     pcd_corr_points = pcd_corr_points[sel_indices]
    #     img_corr_pixels = img_corr_pixels[sel_indices]
    #     pcd_corr_pixels = pcd_corr_pixels[sel_indices]
    #     corr_scores = corr_scores[sel_indices]

    num_correspondences = img_corr_points.shape[0]

    if num_correspondences > 0:
        fine_matching_result_dict = evaluate_correspondences(
            pcd_corr_points, img_corr_points, transform, positive_radius=cfg.eval.acceptance_radius
        )
    else:
        fine_matching_result_dict = {"inlier_ratio": 0.0, "overlap": 0.0, "distance": 0.0}

    inlier_ratio = fine_matching_result_dict["inlier_ratio"]
    overlap = fine_matching_result_dict["overlap"]

    if num_correspondences >= 4:
        intrinsics = data_dict["intrinsics"]
        if planB == False:
            estimated_transform = registration_with_pnp_ransac(
                pcd_corr_points,
                img_corr_pixels,
                intrinsics,
                num_iterations=cfg.ransac.num_iterations,
                distance_tolerance=cfg.ransac.distance_tolerance)
        else:
            estimated_transform = registration_with_pnp_ransac_ours(
                pcd_corr_points,
                img_corr_pixels,
                intrinsics,
                num_iterations=cfg.ransac.num_iterations,
                distance_tolerance=cfg.ransac.distance_tolerance)

        rmse = registration_rmse(pcd_points, transform, estimated_transform)
        registration_recall = float(rmse < cfg.eval.rmse_threshold)
        rre, rte = isotropic_registration_error(transform, estimated_transform)
    else:
        estimated_transform = np.eye(4)
        registration_recall = 0.0

    return inlier_ratio, registration_recall, rmse, rre, rte, estimated_transform

from utils_applications.util_vis import plot_save_poses_self
from utils_applications.camera   import Lie
import matplotlib.pyplot as plt
import open3d as o3d

def vis_odom_plot(gt_transform, pd_transform):
    fig = plt.figure(figsize=(16,8))
    num = len(gt_transform)
    gt_tf_pose = torch.zeros((num,3,4))
    pd_tf_pose = torch.zeros((num,3,4))
    for i in range(num):
        gt_tf  = torch.tensor(gt_transform[i][:3,:4])
        pd_tf  = torch.tensor(pd_transform[i][:3,:4])
        vec_gt = Lie().SE3_to_se3(gt_tf)
        vec_pd = Lie().SE3_to_se3(pd_tf)
        gt_tf_pose[i] = gt_tf
        pd_tf_pose[i] = pd_tf
    plot_save_poses_self(fig, gt_tf_pose, pd_tf_pose)

def vis_odom_open3d(gt_transform, pd_transform, map_pcd):
    num = len(gt_transform)
    for i in range(num):
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.010)
        mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
        pcd_sphere = mesh_sphere.sample_points_uniformly(number_of_points=5000)
        pcd_sphere.transform(np.linalg.inv(gt_transform[i]))
        map_pcd = map_pcd + pcd_sphere

        # if i>=1:
        #     center_i = np.linalg.inv(gt_transform[i])[:3,3]
        #     center_j = np.linalg.inv(gt_transform[i-1])[:3,3]
        #     dx = center_j - center_i
        #     num_lines = 50
        #     pts_lines = np.zeros((num_lines, 3))
        #     rgb_lines = np.zeros((num_lines, 3))
        #     rgb_lines[:,0] = 0.1
        #     rgb_lines[:,1] = 0.1
        #     rgb_lines[:,2] = 0.7
        #     for k in range(num_lines):
        #         pts_lines[k,0] = center_i[0] + k/num_lines*dx[0]
        #         pts_lines[k,1] = center_i[1] + k/num_lines*dx[1]
        #         pts_lines[k,2] = center_i[2] + k/num_lines*dx[2]
        #     line_pcd = o3d.geometry.PointCloud()
        #     line_pcd.points = o3d.utility.Vector3dVector(pts_lines[:,:3])
        #     line_pcd.colors = o3d.utility.Vector3dVector(rgb_lines[:,:3])
        #     map_pcd = map_pcd + line_pcd

        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.010)
        mesh_sphere.paint_uniform_color([0.7, 0.1, 0.1])
        pcd_sphere = mesh_sphere.sample_points_uniformly(number_of_points=5000)
        pcd_sphere.transform(np.linalg.inv(pd_transform[i]))
        map_pcd = map_pcd + pcd_sphere

        # if i>=1:
        #     center_i = np.linalg.inv(pd_transform[i])[:3,3]
        #     center_j = np.linalg.inv(pd_transform[i-1])[:3,3]
        #     dx = center_j - center_i
        #     num_lines = 50
        #     pts_lines = np.zeros((num_lines, 3))
        #     rgb_lines = np.zeros((num_lines, 3))
        #     rgb_lines[:,0] = 0.7
        #     rgb_lines[:,1] = 0.1
        #     rgb_lines[:,2] = 0.1
        #     for k in range(num_lines):
        #         pts_lines[k,0] = center_i[0] + k/num_lines*dx[0]
        #         pts_lines[k,1] = center_i[1] + k/num_lines*dx[1]
        #         pts_lines[k,2] = center_i[2] + k/num_lines*dx[2]
        #     line_pcd = o3d.geometry.PointCloud()
        #     line_pcd.points = o3d.utility.Vector3dVector(pts_lines[:,:3])
        #     line_pcd.colors = o3d.utility.Vector3dVector(rgb_lines[:,:3])
        #     map_pcd = map_pcd + line_pcd
    return map_pcd