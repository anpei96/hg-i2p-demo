from this import d
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
from model_keypoints.match_utils import pairwiseL2Dist, RegularisedTransport

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
            newly added in tr-i2p (tr-align and tr-adapt)
        '''
        self.T0 = torch.eye(4).cuda()
        self.theta0 = torch.zeros((1,6)).cuda()
        self.mlp_a = nn.Linear(6,   32)
        self.mlp_b = nn.Linear(32,   1)
        self.reason_net = MultiHeadAttention(32, 4)
        self.adapt_2d_net = MultiHeadAttention(128, 4)
        self.adapt_3d_net = MultiHeadAttention(128, 4)
        self.msg_decode_2d = nn.Linear(128*2, 128)
        self.msg_decode_3d = nn.Linear(128*2, 128)
        self.msg_decode_2d_tf = MultiHeadAttention(128*2, 4)
        self.msg_decode_3d_tf = MultiHeadAttention(128*2, 4)

        '''
            newly added in tr-i2p (tr-prune)
        '''
        pass

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
        # add 2d/3d center pts
        seg_2d_ct  = data_dict["seg_2d_ct"].detach()
        seg_3d_ct  = data_dict["seg_3d_ct"].detach() # [n, 3]
        
        # processing 2d pts  [m,640]
        us = seg_2d_ct[:,0].long()
        vs = seg_2d_ct[:,1].long()
        # [b,c,h,w] => [c,h,w] => [h,c,w] => [h,w,c]
        fimg_f = img_feats_f[0].transpose(1,0).transpose(2,1)
        fa = fimg_f[vs[:], us[:]]
        seg_2d_fc = fa
        seg_2d_fa = F.normalize(fa, p=2, dim=1)

        # processing 3d pts [n,1152]                
        pts_1x = data_dict["points"][0].detach()  # [m, 3]
        dis_1x = pairwiseL2Dist(seg_3d_ct.unsqueeze(0), pts_1x.unsqueeze(0))[0] # [n, m]
        idx_1x = torch.argmin(dis_1x, dim=1)
        fa = pcd_feats_f[idx_1x[:]]
        seg_3d_fc = fa
        seg_3d_fa = F.normalize(fa, p=2, dim=1)

        '''
            new step 2.2 --- tr-align --- effective
                topological relationships prediction 
                    with heterogeneous graphs reasoning
        '''
        T0 = self.T0 * 1.0
        intrinsics = data_dict["intrinsics"].detach()
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
        # mat_primitive = torch.abs(mat_primitive) # fix a bug of negative number
        output_dict['mat_primitive'] = mat_primitive

        '''
            new step 2.3 --- tr-adapt --- effective
                cross-domain feature adaptation 
                    with topological relationships
        '''
        # part I: feature exchange
        m, c = seg_2d_fc.size()
        n, c = seg_3d_fc.size()
        att_mat_r = mat_primitive # [m,n] this mat is row-normalized
        #   cascade-I exchange some message from the other modality data
        tmp_2d_fc = torch.matmul(att_mat_r, seg_3d_fc)
        tmp_2d_fc = self.adapt_2d_net(seg_2d_fc, tmp_2d_fc, tmp_2d_fc, None)
        tmp_2d_fc = tmp_2d_fc.reshape((m,c))
        tmp_3d_fc = torch.matmul(att_mat_r.t(), seg_2d_fc)
        tmp_3d_fc = self.adapt_3d_net(seg_3d_fc, tmp_3d_fc, tmp_3d_fc, None)
        tmp_3d_fc = tmp_3d_fc.reshape((n,c))
        msg_2d_fd = tmp_2d_fc # [m,c]
        msg_3d_fd = tmp_3d_fc # [n,c]

        # part II: feature updation
        alpha = 0.1
        seg_2d_mask = data_dict["seg_2d_mask"].detach().bool() # [m,h,w]
        seg_3d_mask = data_dict["seg_3d_mask"].detach().bool() # [n,N]
        num_2d = seg_2d_mask.size(0)
        num_3d = seg_3d_mask.size(0)
        #   [1,c,h,w] => [c,h,w] => [h,c,w] => [h,w,c]
        fimg_f = img_feats_f[0].transpose(1,0).transpose(2,1)

        for i in range(num_2d):
            msg_2d_a = fimg_f[seg_2d_mask[i]] # [x,c]
            msg_2d_b = msg_2d_fd[i]           # [c]
            msg_2d_b = msg_2d_b.unsqueeze(0).repeat(msg_2d_a.size(0),1) # [x,c]
            '''
                using tf
            '''
            msg = torch.concat((msg_2d_a, msg_2d_b), dim=1)
            msg = self.msg_decode_2d_tf(msg, msg, msg, None)
            msg = self.msg_decode_2d(msg)
            msg = msg.reshape((-1,c))
            fimg_f[seg_2d_mask[i]] = \
                (1-alpha)*fimg_f[seg_2d_mask[i]] + alpha*msg
        for i in range(num_3d):
            msg_3d_a = pcd_feats_f[seg_3d_mask[i]] 
            msg_3d_b = msg_3d_fd[i]
            msg_3d_b = msg_3d_b.unsqueeze(0).repeat(msg_3d_a.size(0),1)
            '''
                using tf
            '''
            msg = torch.concat((msg_3d_a, msg_3d_b), dim=1)
            msg = self.msg_decode_3d_tf(msg, msg, msg, None)
            msg = self.msg_decode_3d(msg)
            msg = msg.reshape((-1,c))
            pcd_feats_f[seg_3d_mask[i]] = \
                (1-alpha)*pcd_feats_f[seg_3d_mask[i]] + alpha*msg
            # [h,w,c] => [h,c,w] => [c,h,w] => [1,c,h,w]
        img_feats_f = fimg_f.transpose(2,1).transpose(1,0).unsqueeze(0)

        # discard somethings due to the limite gpu memory
        # data_dict.pop("points")
        data_dict.pop("neighbors")
        data_dict.pop("subsampling")
        data_dict.pop("upsampling")
        # print("backbone cost: ", time.time()-start_time)

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
            note-0530
                to make the i2p network stable, we still detach() these features
        '''
        if not self.training:
            ta = time.time()
            output_dict = self.post_process_generate_corres(
                img_feats_c.detach(), pcd_feats_c.detach(), img_feats_f.detach(), pcd_feats_f.detach(), output_dict)
            
            '''
                new step 6.1 --- tr-prune --- 
                    2d-3d correspondences prune 
                        with topological relationships
            '''
            is_tr_prune = True
            if is_tr_prune == True:
                corr_2d = output_dict["img_corr_pixels"].detach()      # [d,2] (h,w)
                corr_2d_ind = output_dict["img_corr_indices"].detach() # [d]   (w+W*h)
                corr_3d = output_dict["pcd_corr_points"].detach()      # [d,3] 
                corr_3d_ind = output_dict["pcd_corr_indices"].detach() # [d]
                corr_sc = output_dict["corr_scores"].detach()          # [d]   (0,1)

                # pre projecting 3d points into 2d pixels with pose prior
                '''
                    todo: t0 needs to be estimated by tr-pnp
                '''
                



                proj_corr_3d = render(corr_3d, intrinsics, T0) # [d,2]
            
                # part I: group correspondences with tr 
                seg_2d_mask = data_dict["seg_2d_mask"].detach().bool() # [m,h,w]
                seg_3d_mask = data_dict["seg_3d_mask"].detach().bool() # [n,N]
                num_2d = seg_2d_mask.size(0)
                num_3d = seg_3d_mask.size(0)
                seg_2d_mask = seg_2d_mask.reshape((num_2d,-1))         # [m,hw]
                relax_mat_c = (mat_primitive>0.5).int().detach()       # [m,n]

                valid_2d_all = (corr_sc*0).bool()
                for i in range(num_2d):
                    _, idx = torch.where(relax_mat_c[i:i+1,:] > 0.8)
                    num_paired = int(idx.size(0))
                    if num_paired == 0:
                        continue
                    m2d = seg_2d_mask[i]
                    m3d = (seg_3d_mask[0,:]*0).bool()
                    for j in range(num_paired):
                        m3d = m3d + seg_3d_mask[idx[j],:]
                    
                    test_2d   = m2d[corr_2d_ind]
                    _, valid_2d = torch.where(test_2d.reshape((1,-1)) > 0.8)
                    corr_3d_i = corr_3d_ind[valid_2d]
                    corr_2d_i = corr_2d_ind[valid_2d]
                    corr_3d_pix = proj_corr_3d[valid_2d]
                    corr_2d_pix = corr_2d[valid_2d]

                    # part II: correspondences pruning 
                    #   Hough voting with homography assumption
                    delta = corr_3d_pix - corr_2d_pix
                    delta = torch.norm(delta, dim=1)
                    keep_idx = (delta <= 8)
                    valid_2d = valid_2d[keep_idx]
                    
                    # summary pruned correspondences
                    valid_2d_all[valid_2d] = True

                    # debug correspondence info.
                    # print(i, "--- idx: ", idx)
                    # print("area: ", torch.sum(m2d)/(640.0*480.0))
                    # print("corr_2d_i: ", corr_2d_i.size())
                    # print("corr_3d_i: ", corr_3d_i.size())
                    # print("")
                
                # part III: update correspondences
                output_dict["img_corr_points"] = \
                    output_dict["img_corr_points"][valid_2d_all]
                output_dict["img_corr_pixels"] = \
                    output_dict["img_corr_pixels"][valid_2d_all]
                output_dict["img_corr_indices"] = \
                    output_dict["img_corr_indices"][valid_2d_all]
                output_dict["pcd_corr_points"] = \
                    output_dict["pcd_corr_points"][valid_2d_all]
                output_dict["pcd_corr_pixels"] = \
                    output_dict["pcd_corr_pixels"][valid_2d_all]
                output_dict["pcd_corr_indices"] = \
                    output_dict["pcd_corr_indices"][valid_2d_all]
                output_dict["corr_scores"] = \
                    output_dict["corr_scores"][valid_2d_all]

                tb = time.time()
                print("==> cost time: ", tb-ta)
                # assert 1==-1

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration
        # print("cost time: ", duration)
        return output_dict

def create_model(cfg):
    model = TopI2PPlus(cfg)
    return model

