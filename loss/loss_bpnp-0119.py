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

def kl_divergence(p, q):
    '''
    ref: https://blog.csdn.net/weixin_52250664/article/details/137565595
    计算两个概率分布之间的KL散度
        : param p: 第一个概率分布，形状为 (batch_size, num_classes)
        : param q: 第二个概率分布，形状为 (batch_size, num_classes)
        : return: KL散度，形状为 (batch_size,)
    '''
    # 避免log(0)出现无穷大
    epsilon = 1e-10
    
    # 计算KL散度，按照公式：D_{KL}(P || Q) = sum(p * log(p/q))
    kl = torch.sum(p * torch.log((p + epsilon) / (q + epsilon)), dim=1)
    return kl

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
        self.pos_radius_2d = cfg.loss.fine_loss.positive_radius_2d
        self.neg_radius_2d = cfg.loss.fine_loss.negative_radius_2d
    
    @torch.no_grad()
    def get_recall(self, gt_corr_mat, fdist_mat):
        # Get feature match recall, divided by number of points which has inlier matches
        num_gt_corr = torch.gt(gt_corr_mat.sum(-1), 0).float().sum() + 1e-12
        src_indices = torch.arange(fdist_mat.shape[0]).cuda()
        src_nn_indices = fdist_mat.min(-1)[1]
        pred_corr_mat = torch.zeros_like(fdist_mat)
        pred_corr_mat[src_indices, src_nn_indices] = 1.0
        recall = (pred_corr_mat * gt_corr_mat).sum() / num_gt_corr
        return recall

    def forward(self, data_dict, output_dict):
        '''
            note-0117
                circle loss + sinkhorn correspondence loss
        '''
        kpts_2d_pix = output_dict["kpts_2d_pix"]
        kpts_3d_pix = output_dict["kpts_3d_pix"]
        kpts_2d_fea = output_dict["kpts_2d_fea"]
        kpts_3d_fea = output_dict["kpts_3d_fea"]

        dist2d_mat = pairwise_distance(
            kpts_2d_pix, kpts_3d_pix, squared=False, strict=True)
        pos_masks = torch.lt(dist2d_mat, self.pos_radius_2d)
        neg_masks = torch.gt(dist2d_mat, self.neg_radius_2d)

        vec_2d_pts = torch.sum(pos_masks, dim=1)
        vec_3d_pts = torch.sum(pos_masks, dim=0)
        ind_2d_pts = (vec_2d_pts >= 1)
        ind_3d_pts = (vec_3d_pts >= 1)
        kpts_2d_pix = kpts_2d_pix[ind_2d_pts,:]
        kpts_3d_pix = kpts_3d_pix[ind_3d_pts,:]

        kpts_2d_fea = output_dict["kpts_2d_fea"][ind_2d_pts]
        kpts_3d_fea = output_dict["kpts_3d_fea"][ind_3d_pts]
        fdist_mat = pairwise_distance(
            kpts_2d_fea, kpts_3d_fea, normalized=False)

        dist2d_mat = pairwise_distance(
            kpts_2d_pix, kpts_3d_pix, squared=False, strict=True)
        pos_masks = torch.lt(dist2d_mat, self.pos_radius_2d)
        neg_masks = torch.gt(dist2d_mat, self.neg_radius_2d)

        loss_match = self.circle_loss(pos_masks, neg_masks, fdist_mat)
        fea_recall = self.get_recall(pos_masks.float(), fdist_mat)
        loss = loss_match 
        # assert 1==-1

        return {"loss": loss, "fea_recall":fea_recall, "loss_match": loss_match}

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
