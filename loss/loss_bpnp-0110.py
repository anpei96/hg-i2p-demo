import torch
import torch.nn as nn

from vision3d.loss import CircleLoss
from vision3d.ops import apply_transform, pairwise_distance, random_choice
from vision3d.ops.metrics import compute_isotropic_transform_error

from model_keypoints.match_utils import pairwiseL2Dist
import model_keypoints.geometry_utilities as geo

import open3d as o3d
import numpy  as np

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

def kpts_learning_analysis(gt_kr_, pd_kr_, th=0.35):
    mask_1x_gt = (gt_kr_[:,0] == 1)
    mask_1x_pd = (pd_kr_[:,0] >= th)
    recall_1x = torch.sum(mask_1x_gt & mask_1x_pd)/torch.sum(mask_1x_gt)
    prec_1x   = torch.sum(mask_1x_gt & mask_1x_pd)/torch.sum(mask_1x_pd)

    mask_3x_gt = (gt_kr_[:,1] == 1)
    mask_3x_pd = (pd_kr_[:,0] >= th)
    recall_3x = torch.sum(mask_3x_gt & mask_3x_pd)/torch.sum(mask_3x_gt)
    prec_3x   = torch.sum(mask_3x_gt & mask_3x_pd)/torch.sum(mask_3x_pd)

    mask_5x_gt = (gt_kr_[:,2] == 1)
    mask_5x_pd = (pd_kr_[:,0] >= th)
    recall_5x = torch.sum(mask_5x_gt & mask_5x_pd)/torch.sum(mask_5x_gt)
    prec_5x   = torch.sum(mask_5x_gt & mask_5x_pd)/torch.sum(mask_5x_pd)

    print("-------------------report-------------------")
    print("  kpts: ", torch.sum(mask_1x_pd), " all pts: ", pd_kr_.size(0))
    print("  recall_1x: ", recall_1x, "prec_1x: ", prec_1x)
    print("  recall_3x: ", recall_3x, "prec_3x: ", prec_3x)
    print("  recall_5x: ", recall_5x, "prec_5x: ", prec_5x)
    print("--------------------------------------------")
    print()

class KeypointsLoss(nn.Module):
    def __init__(self, cfg):
        super(KeypointsLoss, self).__init__()
        self.circle_loss = CircleLoss(
            cfg.loss.fine_loss.positive_margin,
            cfg.loss.fine_loss.negative_margin,
            cfg.loss.fine_loss.positive_optimal,
            cfg.loss.fine_loss.negative_optimal,
            cfg.loss.fine_loss.log_scale)
        self.max_correspondences = cfg.loss.fine_loss.max_correspondences
        self.pos_radius_3d = cfg.loss.fine_loss.positive_radius_3d
        self.neg_radius_3d = cfg.loss.fine_loss.negative_radius_3d
    
    def forward(self, data_dict, output_dict):
        '''
            1. correspondences analysis --- ok 
        '''
        is_need_analysis = True
        if is_need_analysis:
            gt_res = data_dict["points_mask"].detach()
            pd_res = output_dict["coarse_kpts"].detach()
            kpts_learning_analysis(gt_res, pd_res, th=0.40) # 0.35

        '''
            2. get 2d-3d correspondences --- ok

            compared with 3d distance threshold, we find that 2d pixel distance
            threshold is more suitable to correspondences align

            kpts_2d_pts:  torch.Size([3399, 2])
            kpts_3d_pts:  torch.Size([6890, 2])
            c_gt:  torch.Size([1, 3399, 6890])
            c_gt check:  tensor(5595, device='cuda:0')
        '''
        kpts_2d_pix = output_dict["kpts_2d_pix"]          # [m,3] (only used for gt comput)
        kpts_3d_pix = output_dict["kpts_3d_pix_selected"] # [n,3]
        c_gt = pairwiseL2Dist(
            kpts_2d_pix.unsqueeze(0), kpts_3d_pix.unsqueeze(0)) # [m,n]
        # reprojection error less than 3 pixel is seleceted as a correct correspondences
        c_gt = (c_gt <= 3)

        print("c_gt check: ", c_gt.size(), ", ",torch.sum(c_gt)) 
        assert 1==-1

        '''
            3. compute blind pnp loss --- bad (gpu memory limit)
            it consists of two parts: a. 2d-3d correspondence loss
                b. rotation and translation loss
        '''
        postive_mask = (c_gt > 0)
        loss_correspondence = (1-torch.sum(output_dict["P"][postive_mask]))**2
        
        # transform = data_dict["transform"].detach() # [4,4]
        # R_gt = transform[0:3,0:3]
        # t_gt = transform[0:3,3]
        # I = torch.eye(3).cuda()
        # R = geo.angle_axis_to_rotation_matrix(output_dict["theta"][..., :3])[0]
        # t = output_dict["theta"][..., 3:][0]
        # loss_rot = torch.norm(torch.matmul(R, R_gt.t())-I) # [3,3]=>[1]
        # loss_tra = torch.norm(t - t_gt)                    # [3]=>[1]
        
        # beta = 0.1
        # loss_pose = (loss_rot + loss_tra)*beta

        loss = loss_correspondence #+ loss_pose
        return {"loss": loss, "loss_correspondence": loss_correspondence}
        # return {"loss": loss, "loss_correspondence": loss_correspondence, 
        #     "loss_pose": loss_pose}

class EvalFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse_matching(self, output_dict):
        img_length_c = output_dict["img_num_nodes"]
        pcd_length_c = output_dict["pcd_num_nodes"]
        gt_node_corr_min_overlaps = output_dict["gt_node_corr_min_overlaps"]
        gt_img_node_corr_indices = output_dict["gt_img_node_corr_indices"]
        gt_pcd_node_corr_indices = output_dict["gt_pcd_node_corr_indices"]
        img_node_corr_indices = output_dict["img_node_corr_indices"]
        pcd_node_corr_indices = output_dict["pcd_node_corr_indices"]

        masks = torch.gt(gt_node_corr_min_overlaps, self.acceptance_overlap)
        gt_img_node_corr_indices = gt_img_node_corr_indices[masks]
        gt_pcd_node_corr_indices = gt_pcd_node_corr_indices[masks]
        gt_node_corr_mat = torch.zeros(img_length_c, pcd_length_c).cuda()
        gt_node_corr_mat[gt_img_node_corr_indices, gt_pcd_node_corr_indices] = 1.0

        precision = gt_node_corr_mat[img_node_corr_indices, pcd_node_corr_indices].mean()
        return precision

    @torch.no_grad()
    def evaluate_fine_matching(self, data_dict, output_dict):
        transform = data_dict["transform"]
        img_corr_points = output_dict["img_corr_points"]
        pcd_corr_points = output_dict["pcd_corr_points"]
        # only evaluate the correspondences with depth
        corr_masks = torch.gt(img_corr_points[..., -1], 0.0)
        img_corr_points = img_corr_points[corr_masks]
        pcd_corr_points = pcd_corr_points[corr_masks]
        pcd_corr_points = apply_transform(pcd_corr_points, transform)
        corr_distances = torch.linalg.norm(pcd_corr_points - img_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean().nan_to_num_()
        return precision

    @torch.no_grad()
    def evaluate_registration(self, data_dict, output_dict):
        transform = data_dict["transform"]
        est_transform = output_dict["estimated_transform"]
        pcd_points = output_dict["pcd_points"]

        rre, rte = compute_isotropic_transform_error(transform, est_transform)

        realignment_transform = torch.matmul(torch.linalg.inv(transform), est_transform)
        realigned_pcd_points_f = apply_transform(pcd_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_pcd_points_f - pcd_points, dim=1).mean()
        recall = torch.lt(rmse, self.acceptance_rmse).float()
        return rre, rte, rmse, recall

    def forward(self, data_dict, output_dict):
        c_precision = self.evaluate_coarse_matching(output_dict)
        f_precision = self.evaluate_fine_matching(data_dict, output_dict)
        return {"PIR": c_precision, "IR": f_precision}
