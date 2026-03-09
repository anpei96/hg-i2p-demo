import time
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, cast
from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator
from vision3d.layers import ConvBlock, build_act_layer
from vision3d.array_ops import axis_angle_to_rotation_matrix, get_transform_from_rotation_translation
from vision3d.ops import (
    back_project,
    batch_mutual_topk_select,
    create_meshgrid,
    index_select,
    pairwise_cosine_similarity,
    point_to_node_partition,
    render)
from torchvision.transforms import Compose

# isort: split
from .fusion_module  import CrossModalFusionModule
from .image_backbone import FeaturePyramid, ImageBackbone
from .point_backbone import PointBackbone
from .base_model import baseI2P
from .match_utils import pairwiseL2Dist, RegularisedTransport
from .transformer import MultiHeadAttention
from model_keypoints.match_utils import pairwiseL2Dist, RegularisedTransport, ransac_p3p
from model_keypoints.nonlinear_weighted_blind_pnp import NonlinearWeightedBlindPnP
from lietorch import SE3

class TopI2PPlus(baseI2P):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.matching_radius_2d = cfg.model.ground_truth_matching_radius_2d
        self.matching_radius_3d = cfg.model.ground_truth_matching_radius_3d
        self.pcd_num_points_in_patch = cfg.model.pcd_num_points_in_patch

        # fixed for now
        self.img_h_c = 24
        self.img_w_c = 32
        self.img_num_levels_c = 3
        self.overlap_threshold = 0.3
        self.pcd_min_node_size = 5

        self.img_backbone = ImageBackbone(
            cfg.model.image_backbone.input_dim,
            cfg.model.image_backbone.output_dim,
            cfg.model.image_backbone.init_dim,
            dilation=cfg.model.image_backbone.dilation)

        self.pcd_backbone = PointBackbone(
            cfg.model.point_backbone.input_dim,
            cfg.model.point_backbone.output_dim,
            cfg.model.point_backbone.init_dim,
            cfg.model.point_backbone.kernel_size,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
            cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_sigma)
        
        self.transformer = CrossModalFusionModule(
            cfg.model.transformer.img_input_dim,
            cfg.model.transformer.pcd_input_dim,
            cfg.model.transformer.output_dim,
            cfg.model.transformer.hidden_dim,
            cfg.model.transformer.num_heads,
            cfg.model.transformer.blocks,
            use_embedding=cfg.model.transformer.use_embedding)

        self.img_pyramid = FeaturePyramid(cfg.model.transformer.output_dim)
        self.sinkhorn_mu = 0.1
        self.sinkhorn_tolerance=1e-9
        self.sinkhorn = RegularisedTransport(self.sinkhorn_mu, self.sinkhorn_tolerance)

        '''
            newly added in tr-i2p
        '''
        self.T0 = torch.eye(4).cuda()
        self.theta0 = torch.zeros((1,6)).cuda()
        self.mlp_a = nn.Linear(6,   32)
        self.mlp_b = nn.Linear(32,   1)
        self.reason_net = MultiHeadAttention(32, 4)
        self.ransac_p3p = ransac_p3p
        self.wbpnp = NonlinearWeightedBlindPnP()

    def construct_adj_mat(self, a, b, alpha=0.16):
        '''
            a [n,c] tensor
            b [m,c] tensor
        '''
        dis_mat = \
            pairwiseL2Dist(a.unsqueeze(0), b.unsqueeze(0))[0] # [n,m]
        adj_mat = torch.exp(-alpha * dis_mat)            # [n,m]
        return adj_mat

    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."
        torch.cuda.synchronize()
        start_time = time.time()
        output_dict = {}
        
        # 1. Unpack data from data dict
        img_feats, output_dict = self.unpack_2d_3d_data(data_dict, output_dict)
        pcd_feats = data_dict["points_rgb"].detach()

        # 2. Backbone
        '''
            (baseline) backbone totally costs 0.012s
        '''
        img_feats_list = self.img_backbone(img_feats)
        img_feats_x = img_feats_list[-1]  # (B, C8, H/8, W/8), aka, (1, 512, 60, 80)
        img_feats_f = img_feats_list[0]   # (B, C2, H, W), aka, (1, 128, 480, 640)
        pcd_feats_list = self.pcd_backbone(pcd_feats, data_dict)
        pcd_feats_c = pcd_feats_list[-1]  # (Nc, 1024)
        pcd_feats_f = pcd_feats_list[0]   # (Nf, 128)
        
        '''
            new step 2.1
            obtain feature map of each segmentation primitives with coordinates
        '''
        ta = time.time()
        # add 2d/3d center pts
        seg_2d_ct  = data_dict["seg_2d_ct"].detach()
        seg_3d_ct  = data_dict["seg_3d_ct"].detach() # [n, 3]
        
        # processing 2d pts  [m,640]
        us = seg_2d_ct[:,0].long()
        vs = seg_2d_ct[:,1].long()
        # [b,c,h,w] => [c,h,w] => [h,c,w] => [h,w,c]
        fimg_f = img_feats_f[0].transpose(1,0).transpose(2,1)
        fa = fimg_f[vs[:], us[:]]

        # us = (seg_2d_ct[:,0]/8).long()
        # vs = (seg_2d_ct[:,1]/8).long()
        # fimg_x = img_feats_x[0].transpose(1,0).transpose(2,1)
        # fb = fimg_x[vs[:], us[:]]
        # fi  = torch.concat((fa, fb), dim=1)
        # seg_2d_fa = fi
        # seg_2d_fa = fa
        seg_2d_fa = F.normalize(fa, p=2, dim=1)

        # processing 3d pts [n,1152]                
        pts_1x = data_dict["points"][0].detach()  # [m, 3]
        # pts_8x = data_dict["points"][-1].detach() # [m0,3]
        dis_1x = pairwiseL2Dist(seg_3d_ct.unsqueeze(0), pts_1x.unsqueeze(0))[0] # [n, m]
        # dis_8x = pairwiseL2Dist(seg_3d_ct.unsqueeze(0), pts_8x.unsqueeze(0))[0] # [n,m0]
        idx_1x = torch.argmin(dis_1x, dim=1)
        # idx_8x = torch.argmin(dis_8x, dim=1)

        fa = pcd_feats_f[idx_1x[:]]
        # fb = pcd_feats_c[idx_8x[:]]
        # fp  = torch.concat((fa, fb), dim=1)
        # seg_3d_fa = fp 
        # seg_3d_fa = fa
        seg_3d_fa = F.normalize(fa, p=2, dim=1)

        '''
            new step 2.2 --- tr-align 
                topological relationships prediction 
                    with heterogeneous graph representation
        '''
        T0 = self.T0 * 1.0
        intrinsics = data_dict["intrinsics"].detach()
        gt_pose = data_dict["transform"].detach()
        proj_seg_3d_ct = render(seg_3d_ct, intrinsics, T0) # [n,2]
        # exchange coordinates from (v,u) to (u,v)
        tmp = proj_seg_3d_ct[:,0]*1
        proj_seg_3d_ct[:,0] = proj_seg_3d_ct[:,1]
        proj_seg_3d_ct[:,1] = tmp
        proj_seg_3d_ct = proj_seg_3d_ct.float()
        # construct adj matrix ==> heterogeneous graph
        adj_mat_2d_d = self.construct_adj_mat(seg_2d_ct, seg_2d_ct, alpha=0.016) # [m,m]
        adj_mat_2d_f = self.construct_adj_mat(seg_2d_fa, seg_2d_fa, alpha=1.6) # [m,m]
        adj_mat_3d_d = self.construct_adj_mat(proj_seg_3d_ct, proj_seg_3d_ct, alpha=0.016) # [n,n]
        adj_mat_3d_f = self.construct_adj_mat(seg_3d_fa, seg_3d_fa, alpha=1.6) # [n,n]
        adj_mat_2d3d_d = self.construct_adj_mat(seg_2d_ct, proj_seg_3d_ct) # [m,n]
        adj_mat_2d3d_f = self.construct_adj_mat(seg_2d_fa, seg_3d_fa, alpha=1.6) # [m,n]
        zero_mask = (adj_mat_2d3d_d <= 1e-5)
        adj_mat_2d3d_d[zero_mask] = 0
        # matrix nomralization row and colum
        att_mat_2d3d_r = F.normalize(adj_mat_2d3d_d, p=1, dim=1)
        att_mat_2d3d_c = F.normalize(adj_mat_2d3d_d, p=1, dim=0)
        # reasoning from heterogeneous graph
        fad = torch.matmul(adj_mat_2d_d, att_mat_2d3d_c) # [m,n]
        faf = torch.matmul(adj_mat_2d_f, att_mat_2d3d_c) # [m,n]
        fbd = torch.matmul(att_mat_2d3d_r, adj_mat_3d_d) # [m,n]
        fbf = torch.matmul(att_mat_2d3d_r, adj_mat_3d_f) # [m,n]
        fcd = adj_mat_2d3d_d                             # [m,n]
        fcf = adj_mat_2d3d_f                             # [m,n]
        fall = torch.concat((fad.unsqueeze(2), faf.unsqueeze(2),
            fbd.unsqueeze(2), fbf.unsqueeze(2),
            fcd.unsqueeze(2), fcf.unsqueeze(2)), dim=2)  # [m,n,6]
        m, n, c = fall.size()
        fall_vec = fall.reshape(-1,c)                    # [mn,6]
        fall_vec = self.mlp_a(fall_vec)                  # [mn,32]
        fall_vec = self.mlp_b(self.reason_net(fall_vec,fall_vec,fall_vec,None)) # [mn,1]
        mat_primitive = fall_vec.reshape(m,n)            # [m,n]
        mat_primitive = torch.mul(mat_primitive, att_mat_2d3d_r)
        mat_primitive = F.normalize(mat_primitive, p=1, dim=1)
        output_dict['mat_primitive'] = mat_primitive

        '''
            new step 2.3 --- tr-pose 
                pose estimation with predicted topological relationships
                sinkhorn layer does not perform well, 
                so, using M matrix directly
        '''
        if True:
            M = mat_primitive.unsqueeze(0)
            b, m, n = M.size()
            num_points_3d = n
            num_points_2d = m
            '''
                fix a bug about wbpnp layer
            '''
            if ((n > 8) & (m > 8)):
                # computing bearing vector of seg_2d_ct
                m = seg_2d_ct.size(0)
                tmp_vector = torch.ones((m,3)).cuda()
                tmp_vector[:,:2] = seg_2d_ct[:,:2]       # [m,3]
                inv_intr_mat = torch.inverse(intrinsics) # [3,3]
                kpts_2d_bearing = torch.matmul(tmp_vector, inv_intr_mat.t()) # [m,3]

                theta, theta0 = None, None
                p3d = seg_3d_ct.unsqueeze(0)             # [1,n,3]
                p2d = kpts_2d_bearing[:,:2].unsqueeze(0) # [1,m,2]

                # this ransac works :)
                # theta0 = self.ransac_p3p(M, p2d, p3d, num_points_2d, num_points_3d)
                theta0 = self.theta0*1.0 # for the stable training use prior pose

                # Run Weighted BPnP Optimization:
                # it works :)
                p2d_bearings = torch.nn.functional.pad(p2d, (0, 1), "constant", 1.0)
                p2d_bearings = torch.nn.functional.normalize(p2d_bearings, p=2, dim=-1)
                theta = self.wbpnp(M, p2d_bearings, p3d, theta0)
            
                # exchange from (rvec, tvec) to (tvec, rvec) for se(3) mapping
                theta_ = theta*1.0
                tmp = theta_[0,0:3]*1
                theta_[0,0:3] = theta_[0,3:6]
                theta_[0,3:6] = tmp
                transform = SE3.exp(theta_).matrix()[0]
                output_dict['wbpnp_status']  = 1
                output_dict['est_transform'] = transform
                # rvec = theta[0,0:3].detach().cpu().numpy()
                # tvec = theta[0,3:6].detach().cpu().numpy()
                # rotation = axis_angle_to_rotation_matrix(rvec)
                # transform_ = get_transform_from_rotation_translation(rotation, tvec)
            else:
                output_dict['wbpnp_status']  = 0
        
        '''
            new step 2.4 --- project pcd_points_c to proj_pcd_points_c
                using prior pose
        '''
        # B, C, H, W = img_feats_f.size()
        # tmp = render(output_dict["pcd_points_c"], intrinsics, T0) # [n,2]
        # tmp = tmp*1.0
        # tmp[:,0] = tmp[:,0]/H
        # tmp[:,1] = tmp[:,1]/W
        # proj_pcd_points_c = output_dict["pcd_points_c"] * 0
        # proj_pcd_points_c[:,:2] = tmp[:,:2]

        tb = time.time()
        # print("tr-pnp cost time: ", tb-ta)
        # print("transform: ", transform)
        # print("transform_: ", transform_)
        # print("gt_pose: ", gt_pose)
        # print("mat_primitive: ", mat_primitive)
        # print("mat_primitive_gt: ", data_dict["mat_primitive_gt"])
        # assert 1==-1

        # discard somethings due to the limite gpu memory
        # data_dict.pop("points")
        data_dict.pop("neighbors")
        data_dict.pop("subsampling")
        data_dict.pop("upsampling")
        # print("backbone cost: ", time.time()-start_time)

        # 3. Interaction 

        # 3. Transformer
        '''
            (baseline) transformer totally costs 0.080s 
        '''
        # 3.1 Prepare image features
        img_shape_c = (self.img_h_c, self.img_w_c)
        img_feats_c = F.interpolate(img_feats_x, size=img_shape_c, mode="bilinear", align_corners=True)  # to (24, 32)
        img_feats_c = img_feats_c.squeeze(0).view(-1, self.img_h_c * self.img_w_c).transpose(0, 1)       # (768, 512)
        # print("interpolate cost: ", time.time()-start_time)

        # 3.2 Cross-modal fusion transformer
        img_feats_c, pcd_feats_c = self.transformer(
            img_feats_c.unsqueeze(0),
            output_dict["img_pixels_c"].unsqueeze(0),
            pcd_feats_c.unsqueeze(0),
            output_dict["pcd_points_c"].unsqueeze(0))
        # print("transformer cost: ", time.time()-start_time)

        # 3.3 Post-transformer image feature pyramid
        '''
            (baseline) pyramid totally costs 0.002s
        '''
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
        # print("coarse-level matching cost: ", time.time()-start_time)

        '''
            a crucial step to generate 2d-3d corresponding labels
                due to data augmentation in dataloader

            (baseline) it totally costs 0.064s
        '''
        output_dict = self.genenrate_label(output_dict)
        # print("genenrate label cost: ", time.time()-start_time)
       
        # 5. Fine-leval matching
        img_channels_f = img_feats_f.shape[1]
        img_feats_f = img_feats_f.squeeze(0).view(img_channels_f, -1).transpose(0, 1).contiguous()

        img_feats_f = F.normalize(img_feats_f, p=2, dim=1)
        pcd_feats_f = F.normalize(pcd_feats_f, p=2, dim=1)

        output_dict["img_feats_f"] = img_feats_f
        output_dict["pcd_feats_f"] = pcd_feats_f

        # 6. Select topk nearest node correspondences
        '''
            (baseline) all forward procedure totally costs 0.172s
        '''
        if not self.training:
            output_dict = self.post_process_generate_corres(
                img_feats_c.detach(), pcd_feats_c.detach(), img_feats_f.detach(), pcd_feats_f.detach(), output_dict)

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration
        # print("cost time: ", duration)
        return output_dict

def create_model(cfg):
    model = TopI2PPlus(cfg)
    return model

