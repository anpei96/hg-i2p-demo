import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator
from vision3d.ops import (
    back_project,
    batch_mutual_topk_select,
    create_meshgrid,
    index_select,
    pairwise_cosine_similarity,
    point_to_node_partition,
    render)

# isort: split
import numpy as np
import cv2 as cv
import open3d as o3d
from .base_model import baseI2P

import turboreg_gpu
# Initialize TurboReg with specific parameters:
reger = turboreg_gpu.TurboRegGPU(
    6000,      # max_N: Maximum number of correspondences
    0.012,     # tau_length_consis: \tau (consistency threshold for feature length/distance)
    2000,      # num_pivot: Number of pivot points, K_1
    0.15,      # radiu_nms: Radius for avoiding the instability of the solution
    0.1,       # tau_inlier: Threshold for inlier points. NOTE: just for post-refinement (REF@PointDSC/SC2PCR/MAC)
    "IN"       # eval_metric: MetricType (e.g., "IN" for Inlier Number, or "MAE" / "MSE")
)

def pairwiseL2Dist(x1, x2):
    """ Computes the pairwise L2 distance between batches of feature vector sets

    res[..., i, j] = ||x1[..., i, :] - x2[..., j, :]||
    since 
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a^T*b

    Adapted to batch case from:
        jacobrgardner
        https://github.com/pytorch/pytorch/issues/15253#issuecomment-491467128
    """
    x1_norm2 = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm2 = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.baddbmm(
        x2_norm2.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm2).clamp_min_(1e-30).sqrt_()
    return res

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

class baseline_with_prune(baseI2P):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg

    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."
        torch.cuda.synchronize()
        start_time = time.time()
        output_dict = {}
        
        # 1. Unpack data from data dict
        image, output_dict = self.unpack_2d_3d_data(data_dict, output_dict)
        pcd_feats = data_dict["points_rgb"].detach()

        # 2. Backbone
        img_feats_list = self.img_backbone(image)
        img_feats_x = img_feats_list[-1]  # (B, C8, H/8, W/8), aka, (1, 512, 60, 80)
        img_feats_f = img_feats_list[0]   # (B, C2, H, W), aka, (1, 128, 480, 640)

        pcd_feats_list = self.pcd_backbone(pcd_feats, data_dict)
        pcd_feats_c = pcd_feats_list[-1]  # (Nc, 1024)
        pcd_feats_f = pcd_feats_list[0]   # (Nf, 128)

        # discard somethings due to the limite gpu memory
        # data_dict.pop("points")
        data_dict.pop("neighbors")
        data_dict.pop("subsampling")
        data_dict.pop("upsampling")

        # 3. Transformer
        # 3.1 Prepare image features
        img_shape_c = (self.img_h_c, self.img_w_c)
        img_feats_c = F.interpolate(img_feats_x, size=img_shape_c, mode="bilinear", align_corners=True)  # to (24, 32)
        img_feats_c = img_feats_c.squeeze(0).view(-1, self.img_h_c * self.img_w_c).transpose(0, 1)       # (768, 512)

        # 3.2 Cross-modal fusion transformer
        img_feats_c, pcd_feats_c = self.transformer(
            img_feats_c.unsqueeze(0),
            output_dict["img_pixels_c"].unsqueeze(0),
            pcd_feats_c.unsqueeze(0),
            output_dict["pcd_points_c"].unsqueeze(0),
        )

        # 3.3 Post-transformer image feature pyramid
        img_feats_c = img_feats_c.transpose(1, 2).contiguous().view(1, -1, self.img_h_c, self.img_w_c)
        all_img_feats_c = self.img_pyramid(img_feats_c)
        all_img_feats_c = [x.squeeze(0).view(x.shape[1], -1).transpose(0, 1).contiguous() for x in all_img_feats_c]
        img_feats_c = torch.cat(all_img_feats_c, dim=0)

        # 4. Coarse-level matching
        pcd_feats_c = pcd_feats_c.squeeze(0)
        img_feats_c = F.normalize(img_feats_c, p=2, dim=1)
        pcd_feats_c = F.normalize(pcd_feats_c, p=2, dim=1)

        output_dict["img_feats_c"] = img_feats_c
        output_dict["pcd_feats_c"] = pcd_feats_c
        output_dict = self.genenrate_label(output_dict)
       
        # 5. Fine-leval matching
        img_channels_f = img_feats_f.shape[1]
        img_feats_f = img_feats_f.squeeze(0).view(img_channels_f, -1).transpose(0, 1).contiguous()

        img_feats_f = F.normalize(img_feats_f, p=2, dim=1)
        pcd_feats_f = F.normalize(pcd_feats_f, p=2, dim=1)

        output_dict["img_feats_f"] = img_feats_f
        output_dict["pcd_feats_f"] = pcd_feats_f

        # 6. Select topk nearest node correspondences
        if not self.training:
            output_dict = self.post_process_generate_corres(
                img_feats_c.detach(), pcd_feats_c.detach(), img_feats_f.detach(), pcd_feats_f.detach(), output_dict)

            '''
                note-0717 add a module of i2p-pruning 
                    using depths predicted from depthanything v2 (metric-depth)
            '''
            time_begin = time.time()
            corr_2d = output_dict["img_corr_pixels"].detach()       # [d,2] (h,w)
            corr_2d_ind = output_dict["img_corr_indices"].detach()  # [d]   (w+W*h)
            corr_3d = output_dict["pcd_corr_points"].detach()       # [d,3] 
            corr_sc = output_dict["corr_scores"].detach()           # [d]   (0,1)
            intrinsics = data_dict["intrinsics"].detach()           # [3,3]
            intr_inv_t = torch.linalg.inv(intrinsics).transpose(1,0)
            
            # in the debug stage I,  we use the gt depth
            # depth_pred = data_dict["depth"].detach()                # [H,W]
            # in the debug stage II, we use the pred depth
            '''
                note-0802
                    pred depth is a affine-invarient depth
            '''
            depth_pred = data_dict["depth_pred"].detach()           # [H,W]
            depth_vec  = depth_pred.reshape((-1))                   # [HW]

            '''
                step 1 generates vectors w and v [d,d,3]
            '''
            # step 1.1 generate vectors of v
            num_cor = corr_2d.size(0)
            temp_p1 = corr_3d.repeat((num_cor,1,1)) # [d,d,3]
            temp_p2 = temp_p1.transpose(1,0)        # [d,d,3]
            v_colls = temp_p1 - temp_p2          

            # step 1.2 generate vectos of w
            d_vec = depth_vec[corr_2d_ind].reshape((-1,1)) # [d,1]
            corr_2d_a = torch.ones((num_cor,3)).cuda()
            corr_2d_a[:,0] = corr_2d[:,1]
            corr_2d_a[:,1] = corr_2d[:,0]
            temp_ww = torch.matmul(corr_2d_a, intr_inv_t) # [d,3]
            dd_vec  = torch.concat((d_vec, d_vec, d_vec), dim=1) # [d,3]
            
            corr_3dm = torch.mul(dd_vec, temp_ww)    # [d,3]
            temp_p1 = corr_3dm.repeat((num_cor,1,1)) # [d,d,3]
            temp_p2 = temp_p1.transpose(1,0)         # [d,d,3]
            w_colls = temp_p1 - temp_p2   

            '''
                step 2 scale-rotation-inliers loop optimization
            '''
            # step 2.1 scale estimation
            #   also this sub-step is used for check w and v correctness
            v_colls_norm = \
                torch.norm(v_colls, dim=2, keepdim=True) # [d,d,1]
            w_colls_norm = \
                torch.norm(w_colls, dim=2, keepdim=True) # [d,d,1]
            
            # if scale is unique (just in the ideal case)
            scale_colls = (v_colls_norm/w_colls_norm)    # [d,d,1]
            scales = scale_colls[:,:,0]
            scales = \
                torch.where(torch.isnan(scales), torch.full_like(scales, -1), scales)
            scales = \
                torch.where(torch.isinf(scales), torch.full_like(scales, -1), scales)
            scales = \
                torch.where(scales==0          , torch.full_like(scales, -1), scales)

            # remove scales beyond 0.1
            max_scale  = 0.01
            mask_scale = scales > max_scale
            scales[mask_scale] = -1

            # zoom it
            scales = scales * 1000.0

            # histogram can only be used in cpu mode
            max_bins = 1000*10
            hist_scales = scales.cpu().histogram(bins=max_bins,range=(0, torch.max(scales)))
            visual_hist = hist_scales.hist.reshape((-1)).cuda()
            visual_bins = hist_scales.bin_edges.reshape((-1)).cuda()

            # scale estimated by the peak of histogram
            idx_max = torch.argmax(visual_hist)
            sca_opt = visual_bins[idx_max] / 1000.0

            # re-scale point cloud generated by depth anything v2 --- ok
            corr_3dm = corr_3dm*sca_opt
            
            # pts_b = corr_3dm.cpu().detach().numpy()
            # pts_a = corr_3d .cpu().detach().numpy()
            # cor_b = np.zeros_like(pts_b)
            # cor_a = np.zeros_like(pts_a)
            # cor_b[:,0] = 1
            # cor_a[:,1] = 1
            # pcda = o3d.geometry.PointCloud()
            # pcda.points = o3d.utility.Vector3dVector(pts_a[:,:3])
            # pcda.colors = o3d.utility.Vector3dVector(cor_a[:,:3])
            # pcdb = o3d.geometry.PointCloud()
            # pcdb.points = o3d.utility.Vector3dVector(pts_b[:,:3])
            # pcdb.colors = o3d.utility.Vector3dVector(cor_b[:,:3])
            # show_pcd(pcda+pcdb)

            '''
                scheme-0 naive approach
            '''
            is_used_scheme_0 = True
            if is_used_scheme_0:
                sca_max = visual_bins[idx_max]
                max_res = 500
                scale_a = sca_max - torch.max(scales)/max_bins * max_res
                scale_b = sca_max + torch.max(scales)/max_bins * max_res
                valid_mask = (scale_colls >= scale_a/1000) \
                    & (scale_colls <= scale_b/1000)
                scale_colls[~valid_mask] = 0 
                scale_colls[valid_mask]  = 1
                map_scale = torch.sum(scale_colls, dim=1)
                valid_mask = (map_scale > torch.mean(map_scale)) # torch.mean(map_scale)
                valid_mask = valid_mask.reshape((-1))

            '''
                scheme-1 turboreg iccv25
            '''
            is_used_scheme_1 = False
            if is_used_scheme_1:
                kpts_src = corr_3dm # [n,3]
                kpts_dst = corr_3d  # [n,3]
                # run registration
                trans = reger.run_reg(kpts_src, kpts_dst).cuda()
                res_R = trans[0:3,0:3] # [3,3]
                res_T = trans[0:3,3:4] # [3,1]
                # pruning
                tmp = kpts_src.T # [3,n]
                tmp = torch.matmul(res_R, tmp)
                tmp = tmp + res_T
                tmp = tmp.T
                dis = torch.norm(kpts_dst-tmp, dim=1, keepdim=False)
                valid_mask = (dis <= 0.20) # 0.10
            
            '''
                scheme-2 mac++ cvpr24
            '''
            
            
            time_end = time.time()
            print("=> pruning time cost: ", time_end - time_begin)

            # === visualize histogram --- ok === #
            # import matplotlib.pyplot as plt
            # visual_hist = hist_scales.hist.detach().numpy().reshape((-1))
            # visual_bins = hist_scales.bin_edges.detach().numpy().reshape((-1))
            # vis_bins = max_bins
            # plt.plot(visual_bins[:vis_bins], visual_hist[:vis_bins])
            # plt.show()

            # for debug usage
            # transform = data_dict["transform"].detach()
            # print("transform: ", transform)
            # print("l2_mask: ", l2_mask.size())
            # print("dc_mask: ", dc_mask.size())
            # assert 1==-1
            
            '''
                step 3 update correspondences
            '''
            output_dict["img_corr_points"] = \
                output_dict["img_corr_points"][valid_mask]
            output_dict["img_corr_pixels"] = \
                output_dict["img_corr_pixels"][valid_mask]
            output_dict["img_corr_indices"] = \
                output_dict["img_corr_indices"][valid_mask]
            output_dict["pcd_corr_points"] = \
                output_dict["pcd_corr_points"][valid_mask]
            output_dict["pcd_corr_pixels"] = \
                output_dict["pcd_corr_pixels"][valid_mask]
            output_dict["pcd_corr_indices"] = \
                output_dict["pcd_corr_indices"][valid_mask]
            output_dict["corr_scores"] = \
                output_dict["corr_scores"][valid_mask]

            '''
                step 3.1 using ransac for the post-proc
            '''
            is_need_ransac = False
            if is_need_ransac:
                corr_2d = output_dict["img_corr_pixels"].detach()       # [d,2] (h,w)
                corr_3d = output_dict["pcd_corr_points"].detach()       # [d,3] 
                intrinsics = data_dict["intrinsics"].detach()           # [3,3]
                ct_3d = corr_3d.detach().cpu().numpy()
                ct_2d = corr_2d.detach().cpu().numpy()
                k_mat = intrinsics.detach().cpu().numpy()
                ct_2d_ = ct_2d * 1.0
                ct_2d_[:,0] = ct_2d[:,1]
                ct_2d_[:,1] = ct_2d[:,0]
                success, rvec, tvec, inliers = \
                    cv.solvePnPRansac(ct_3d, ct_2d_, k_mat, None,
                    iterationsCount=1000, reprojectionError=10, flags=cv.SOLVEPNP_P3P)
                valid_mask = inliers.reshape((-1))

                output_dict["img_corr_points"] = \
                    output_dict["img_corr_points"][valid_mask]
                output_dict["img_corr_pixels"] = \
                    output_dict["img_corr_pixels"][valid_mask]
                output_dict["img_corr_indices"] = \
                    output_dict["img_corr_indices"][valid_mask]
                output_dict["pcd_corr_points"] = \
                    output_dict["pcd_corr_points"][valid_mask]
                output_dict["pcd_corr_pixels"] = \
                    output_dict["pcd_corr_pixels"][valid_mask]
                output_dict["pcd_corr_indices"] = \
                    output_dict["pcd_corr_indices"][valid_mask]
                output_dict["corr_scores"] = \
                    output_dict["corr_scores"][valid_mask]

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration
        return output_dict

def create_model(cfg):
    model = baseline_with_prune(cfg)
    return model

