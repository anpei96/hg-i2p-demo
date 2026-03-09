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

from .gconstructor import GraphConstructorFor3DMatch, GraphConstructorForI2PReg
from .gfilter import graphFilter,datasample
config_fastmac={
    "num_points":6000,
    "resolution":0.006,
    "data_dir":'/data/Processed_3dmatch_3dlomatch/',
    "name":"3dmatch",
    'descriptor':'fpfh',
    'batch_size':1,
    'inlier_thresh':0.2,
    'device':'cuda',
    'mode':'graph',
    'ratio':0.50,}
gc     = GraphConstructorFor3DMatch()
gc_i2p = GraphConstructorForI2PReg()

import json
from easydict import EasyDict as edict
from .SC2_PCR import Matcher
from .SC2_PCR_plus import Matcher_plus
from .mac import prune_pipeline

def normalize(x):
    # transform x to [0,1]
    x=x-x.min()
    x=x/x.max()
    return x

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
            scale_colls_raw = scale_colls * 1
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
            max_bins = 1000*10#00
            hist_scales = scales.cpu().histogram(bins=max_bins,range=(0, torch.max(scales)))
            visual_hist = hist_scales.hist.reshape((-1)).cuda()
            visual_bins = hist_scales.bin_edges.reshape((-1)).cuda()

            # scale estimated by the peak of histogram
            idx_max = torch.argmax(visual_hist)
            sca_opt = visual_bins[idx_max] / 1000.0

            # re-scale point cloud generated by depth anything v2 --- ok
            '''
                note-0818
                    need to sample the possible scales, like that
                    +0.05 +0.10 +0.15
                    -0.05 -0.10 -0.15
            '''
            corr_3dm_raw = corr_3dm * 1.0
            corr_3dm = corr_3dm*sca_opt #(sca_opt+0.05/ 1000.0)
            
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
            # print("==> the best scale: ", visual_bins[idx_max])
            # show_pcd(pcda+pcdb)
            # assert 1==-1

            # === visualize histogram --- ok === #
            # import matplotlib.pyplot as plt
            # visual_hist = hist_scales.hist.detach().numpy().reshape((-1))
            # visual_bins = hist_scales.bin_edges.detach().numpy().reshape((-1))
            # vis_bins = max_bins
            # plt.plot(visual_bins[:vis_bins], visual_hist[:vis_bins])
            # plt.show()
            # assert 1==-1

            # default setting
            valid_mask = (d_vec[:,0] >= -1)

            '''
                scheme-0 naive approach
            '''
            is_used_scheme_0 = False
            # is_used_scheme_0 = True
            if is_used_scheme_0:
                sca_max = visual_bins[idx_max]
                max_res = 100
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
                scheme-0-plus naive approach++
            '''
            is_used_scheme_0pp = False
            # is_used_scheme_0pp = True
            if is_used_scheme_0pp:
                num_topk = 5
                inf_topk = torch.topk(visual_hist, num_topk)
                sca_cans = visual_bins[inf_topk.indices]
                max_res  = 0.5
                scale_as = sca_cans - torch.max(scales)/max_bins * max_res
                scale_bs = sca_cans + torch.max(scales)/max_bins * max_res
                valid_mask = torch.zeros_like(scale_colls).bool()
                for i in range(num_topk):
                    valid_mask_ = (scale_colls >= scale_as[i]/1000) \
                        & (scale_colls <= scale_bs[i]/1000)
                    valid_mask = valid_mask | valid_mask_
                
                scale_colls[~valid_mask] = 0 
                scale_colls[valid_mask]  = 1
                map_scale = torch.sum(scale_colls, dim=1)
                valid_mask = (map_scale > torch.mean(map_scale)) # torch.mean(map_scale)
                valid_mask = valid_mask.reshape((-1))
                print("scale_as: ", scale_as)
                print("scale_bs: ", scale_bs)

            '''
                scheme-1 turboreg iccv25
            '''
            is_used_scheme_1 = False
            # is_used_scheme_1 = True
            if is_used_scheme_1:
                kpts_src = corr_3dm # [n,3]
                kpts_dst = corr_3d  # [n,3]
                # run registration
                trans = reger.run_reg(kpts_src, kpts_dst).cuda()
                res_R = trans[0:3,0:3] # [3,3]
                res_T = trans[0:3,3:4] # [3,1]
                print("==> turboreg se(3) error")
                print(" rot error: ", torch.norm((res_R-torch.eye(3).cuda())))
                print(" tra error: ", torch.norm(res_T))
                # pruning
                tmp = kpts_src.T # [3,n]
                tmp = torch.matmul(res_R, tmp)
                tmp = tmp + res_T
                tmp = tmp.T
                dis = torch.norm(kpts_dst-tmp, dim=1, keepdim=False)
                # valid_mask = (dis <= 0.20) # 0.10
                
                # for scannet experiments only
                valid_mask = (dis <= 0.20) # 0.10
                # assert 1==-1
            
            '''
                scheme-2 fast-mac cvpr24
            '''
            is_used_scheme_2 = False
            # is_used_scheme_2 = True
            if is_used_scheme_2:
                kpts_src = corr_3dm # [n,3]
                kpts_dst = corr_3d  # [n,3]
                inputs_c = torch.concatenate((kpts_src, kpts_dst), dim=1)
                inputs_c = inputs_c.unsqueeze(0) # [1,n,6]
                corr_graph=gc(
                    inputs_c, 
                    config_fastmac["resolution"], 
                    config_fastmac["name"], 
                    config_fastmac["descriptor"], 
                    config_fastmac["inlier_thresh"])
                # print("inputs_c: ", inputs_c)
                # print("corr_graph: ", corr_graph)
                degree_signal = torch.sum(corr_graph,dim=-1)
                corr_laplacian = \
                    (torch.diag_embed(degree_signal)-corr_graph).squeeze(0)
                corr_scores = graphFilter(
                    degree_signal.transpose(0,1), 
                    corr_laplacian, is_sparse=False)
                corr_scores  = normalize(corr_scores)
                total_scores = corr_scores
                sample_ratio = config_fastmac["ratio"]
                k    = int(inputs_c.shape[1]*sample_ratio)
                idxs = datasample(k, False, total_scores)
                valid_mask = idxs

            '''
                scheme-3 sc2-pcr cvpr22
            '''
            is_used_scheme_3 = False
            # is_used_scheme_3 = True
            if is_used_scheme_3:
                config_path = \
                    "/media/anpei/DiskA/05_i2p_fewshot/model_matr/config_json/" + \
                    "config_3DMatch.json"
                config  = json.load(open(config_path, 'r'))
                config  = edict(config)
                matcher = Matcher(
                    inlier_threshold=config.inlier_threshold,
                    num_node=config.num_node,
                    use_mutual=config.use_mutual,
                    d_thre=config.d_thre,
                    num_iterations=config.num_iterations,
                    ratio=config.ratio,
                    nms_radius=config.nms_radius,
                    max_points=config.max_points,
                    k1=config.k1,
                    k2=config.k2,)
                
                kpts_src = corr_3dm.unsqueeze(0) # [1,n,3]
                kpts_dst = corr_3d.unsqueeze(0)  # [1,n,3]
                pred_trans, pred_labels, src_keypts_corr, tgt_keypts_corr = \
                    matcher.estimator(
                        kpts_src, kpts_dst, None, None)
                valid_mask = pred_labels.reshape((-1)).bool()
            
            '''
                scheme-4 sc2-pcr++ tpami23
            '''
            is_used_scheme_4 = False
            # is_used_scheme_4 = True
            if is_used_scheme_4:
                config_path = \
                    "/media/anpei/DiskA/05_i2p_fewshot/model_matr/config_json/" + \
                    "config_3DMatch.json"
                config  = json.load(open(config_path, 'r'))
                config  = edict(config)
                matcher = Matcher_plus(
                    inlier_threshold=config.inlier_threshold*8,
                    num_node=config.num_node,
                    use_mutual=config.use_mutual,
                    d_thre=config.d_thre,
                    num_iterations=config.num_iterations,
                    ratio=config.ratio,
                    nms_radius=config.nms_radius,
                    max_points=config.max_points,
                    k1=config.k1,
                    k2=config.k2,)
                kpts_src = corr_3dm.unsqueeze(0) # [1,n,3]
                kpts_dst = corr_3d.unsqueeze(0)  # [1,n,3]
                pred_trans, pred_labels, src_keypts_corr, tgt_keypts_corr = \
                    matcher.estimator(
                        kpts_src, kpts_dst, kpts_src, kpts_dst)
                valid_mask = pred_labels.reshape((-1)).bool()
            
            '''
                scheme-5 mac cvpr23
            '''
            is_used_scheme_5 = False
            # is_used_scheme_5 = True
            if is_used_scheme_5:
                kpts_src = corr_3dm # [n,3]
                kpts_dst = corr_3d  # [n,3]
                inputs_c = torch.concatenate((kpts_src, kpts_dst), dim=1)
                # inputs_c = inputs_c.unsqueeze(0) # [1,n,6]
                
                trans = prune_pipeline(inputs_c)
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
                scheme-6 new spatial compatibility
            '''
            is_used_scheme_6 = False
            is_used_scheme_6 = True
            if is_used_scheme_6:
                # step 1. uniformly sampling candidate scales
                sca_opt_1k = visual_bins[idx_max]
                scales = scale_colls_raw[:,:,0] # [n,n]
                scales = \
                    torch.where(torch.isnan(scales), torch.full_like(scales, 0), scales)
                scales = \
                    torch.where(torch.isinf(scales), torch.full_like(scales, 0), scales)
                scales_1k = scales * 1e3        # [n,n]
                basis     = torch.arange(-10,11).cuda() * 0.01 # 0.01 0.05 in scannet+tum
                sca_samples_1k = sca_opt_1k + basis # [b]
                num_samples    = int(basis.size(0))
                param_n        = int(scales_1k.size(0))

                # step 2. recover candidate correspondences with optimal scales
                kpts_src     = corr_3dm_raw   # [n,3]
                kpts_dst     = corr_3d        # [n,3]
                sample_ratio = config_fastmac["ratio"]
                sample_ratio = 0.50           # 0.60 #0.50
                # step 2.0. variables preparation
                k            = int(param_n*sample_ratio)
                kpts_src_co  = torch.zeros((num_samples,k,3)).cuda()
                kpts_dst_co  = torch.zeros((num_samples,k,3)).cuda()
                de_graph_fog = torch.zeros((num_samples,param_n,param_n)).cuda()
                de_graph_sog = torch.zeros((num_samples,param_n,param_n)).cuda()
                valid_mask_all = []

                for id in range(num_samples):
                    # a trick to filter out some unreliable scales
                    # id = num_samples // 2
                    if id <= num_samples // 2 - 4:
                        id = num_samples // 2
                    if id >= num_samples // 2 + 4:
                        id = num_samples // 2
                    # id = num_samples // 2

                    # step 2.1. focus on graph with closed scales
                    sca_cand         = sca_samples_1k[id]
                    msk_cand         = \
                        (scales_1k >= sca_cand-0.004) & (scales_1k <= sca_cand+0.004)
                    de_graph_fog[id] = msk_cand
                    
                    # step 2.2. compute the second-order graph
                    msk_cand = msk_cand.float()
                    msk_cand = msk_cand*torch.matmul(msk_cand, msk_cand.T)
                    de_graph_sog[id] = msk_cand

                    # step 2.3. approximate search cliques via graph filter
                    degree_signal  = torch.sum(de_graph_sog[id],dim=-1)
                    corr_laplacian = \
                        (torch.diag_embed(degree_signal)-de_graph_sog[id])
                    corr_scores = graphFilter(
                        degree_signal.reshape((-1,1)), 
                        corr_laplacian, is_sparse=False)
                    if torch.sum(corr_scores) > 1e-3:
                        corr_scores = normalize(corr_scores)
                    else:
                        corr_scores[:] = 1.0/param_n
                    total_scores = corr_scores
                    idxs = datasample(k, False, total_scores)

                    # step 2.4 collect candidate correspondences with optimal scales
                    kpts_src_co[id] = kpts_src[idxs] * sca_cand/1e3
                    kpts_dst_co[id] = kpts_dst[idxs]
                    valid_mask_all.extend(idxs)
                    # break
                # assert 1==-1

                # step 3. 3d registration with 
                #            candidate correspondences with optimal scales
                # step 3.1. adopt turboreg for initial registration
                #           --- it is better than turboreg :)
                kpts_src_as = kpts_src_co.reshape((-1,3))
                kpts_dst_as = kpts_dst_co.reshape((-1,3))
                trans = reger.run_reg(kpts_src_as, kpts_dst_as).cuda()
                res_R = trans[0:3,0:3] # [3,3]
                res_T = trans[0:3,3:4] # [3,1]
                print("==> ours se(3) error")
                print(" rot error: ", torch.norm((res_R-torch.eye(3).cuda())))
                print(" tra error: ", torch.norm(res_T))

                # step 3.2. pruning correspondences with initial guess
                #           --- project 3d pts into 2d image plane
                corr_2d   = output_dict["img_corr_pixels"].detach() # [d,2] (h,w)
                corr_3d   = output_dict["pcd_corr_points"].detach() # [d,3] 
                pose      = torch.linalg.inv(trans)
                corr_2d_r = render(corr_3d, intrinsics, pose)
                dis = torch.norm(corr_2d_r-corr_2d, dim=1, keepdim=False)
                valid_mask = (dis <= 16.0) # 16 is other 8 is scannet+tum

                # pruning --- planning b (used in scannet) not good
                # kpts_src     = corr_3dm   # [n,3]
                # tmp = kpts_src.T # [3,n]
                # tmp = torch.matmul(res_R, tmp)
                # tmp = tmp + res_T
                # tmp = tmp.T
                # # print("kpts_dst: ", kpts_dst)
                # # print("kpts_src: ", tmp)
                # dis = torch.norm(kpts_dst-tmp, dim=1, keepdim=False)
                # valid_mask = (dis <= 0.20) # 0.10

                # step 3.x. check the collected candidates --- ok 
                # kpts_src_as = kpts_src_co.reshape((-1,3))
                # kpts_dst_as = kpts_dst_co.reshape((-1,3))
                # print("kpts_src_as: ", kpts_src_as.size())
                # pts_b = kpts_src_as.cpu().detach().numpy()
                # pts_a = kpts_dst_as.cpu().detach().numpy()
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
                # assert 1==-1

            time_end = time.time()
            print("=> pruning time cost: ", time_end - time_begin)
            
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
                # print("rvec: ", rvec)
                # print("tvec: ", tvec)
                from vision3d.array_ops import axis_angle_to_rotation_matrix, get_transform_from_rotation_translation
                rotation = axis_angle_to_rotation_matrix(rvec[:,0])
                T_opt = get_transform_from_rotation_translation(rotation, tvec[:,0])
                print("T_opt: ", T_opt)
                print("inliers: ", inliers.shape)

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

