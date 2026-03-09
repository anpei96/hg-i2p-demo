import time
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, cast
from vision3d.models.geotransformer import SuperPointMatchingMutualTopk, SuperPointProposalGenerator
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

class fewshotI2P(baseI2P):
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

    def mask_generate_from_seg(self, image_seg, point_seg):
        # check point seg visulization --- ok 
        # import open3d as o3d
        # from .base_model import show_pcd 
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(
        #     output_dict["pcd_points_f"][:,:3].detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(
        #     point_seg[:,:3].detach().cpu().numpy())
        # show_pcd(pcd)

        image_seg = (image_seg * 255).int() # [H,W,3]
        point_seg = (point_seg * 255).int() # [N,3]
        image_seg_v = image_seg[:,:,0]*1e6 + image_seg[:,:,1]*1e3 + image_seg[:,:,2]
        point_seg_v = point_seg[:,0]*1e6 + point_seg[:,1]*1e3 + point_seg[:,2]
        i_a, i_b = torch.unique(image_seg_v, return_inverse=True)
        p_a, p_b = torch.unique(point_seg_v, return_inverse=True)
        num_img_seg, num_pts_seg = i_a.size(0), p_a.size(0)
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
                    continue # detect black region
            mask_i = (i_b == i)
            img_seg_mask.append(mask_i)
        for j in range(num_pts_seg):
            mask_j = (p_b == j)
            pts_seg_mask.append(mask_j)
        if is_detect_black == True:
            num_img_seg = num_img_seg - 1
        # print("point_seg: ", point_seg)
        # print("image_seg_v: ", image_seg_v.size())
        # print("point_seg_v: ", point_seg_v.size())
        # print("i_a, i_b: ", i_a.size(), i_b.size())
        # print("p_a, p_b: ", p_a.size(), p_b.size())
        return num_img_seg, num_pts_seg, img_seg_mask, pts_seg_mask

    def fea_generate_from_mask(self, 
        img_feats_f, pcd_feats_f, img_corrds, pts_corrds,
        num_img_seg, num_pts_seg, img_seg_mask, pts_seg_mask):
        tmp = img_feats_f.transpose(0,2).transpose(1,3) # (480,128,1,640) => (480,640,1,128)
        tmp = tmp.squeeze(2)
        img_fea_delt_sels = []
        img_fea_mean_sels = []
        img_cor_delt_sels = []
        img_cor_mean_sels = []
        for i in range(num_img_seg):
            mask = img_seg_mask[i]
            img_fea_sel = tmp[mask]
            img_cor_sel = img_corrds[mask]
            img_fea_sel_mean = torch.mean(img_fea_sel, dim=0)
            img_cor_sel_mean = torch.mean(img_cor_sel, dim=0)
            img_fea_delt_sels.append(img_fea_sel - img_fea_sel_mean)
            img_cor_delt_sels.append(img_cor_sel - img_cor_sel_mean)
            img_fea_mean_sels.append(img_fea_sel_mean)
            img_cor_mean_sels.append(img_cor_sel_mean)
            # print("i-th: ", img_fea_sel.size())
        pts_fea_delt_sels = []
        pts_fea_mean_sels = []
        pts_cor_delt_sels = []
        pts_cor_mean_sels = []
        for j in range(num_pts_seg):
            mask = pts_seg_mask[j]
            pts_fea_sel = pcd_feats_f[mask]
            pts_cor_sel = pts_corrds[mask]
            pts_fea_sel_mean = torch.mean(pts_fea_sel, dim=0)
            pts_cor_sel_mean = torch.mean(pts_cor_sel, dim=0)
            pts_fea_delt_sels.append(pts_fea_sel - pts_fea_sel_mean)
            pts_cor_delt_sels.append(pts_cor_sel - pts_cor_sel_mean)
            pts_fea_mean_sels.append(pts_fea_sel_mean)
            pts_cor_mean_sels.append(pts_cor_sel_mean)
            # print("j-th: ", pts_fea_sel.size())
        return img_fea_delt_sels, pts_fea_delt_sels, img_cor_delt_sels, pts_cor_delt_sels, \
               img_fea_mean_sels, pts_fea_mean_sels, img_cor_mean_sels, pts_cor_mean_sels

    def forward(self, data_dict):
        assert data_dict["batch_size"] == 1, "Only batch size of 1 is supported."
        torch.cuda.synchronize()
        start_time = time.time()
        output_dict = {}
        
        # 1. Unpack data from data dict
        img_feats, output_dict = self.unpack_2d_3d_data(data_dict, output_dict)
        pcd_feats = data_dict["points_rgb"].detach()
        '''
            load image and point cloud segmentation results and preprocess them
        '''
        image_seg = data_dict["seg_img_a"].detach()  # [H,W,3]
        point_seg = data_dict["points_seg"].detach() # [N,3]
        num_img_seg, num_pts_seg, img_seg_mask, pts_seg_mask = \
            self.mask_generate_from_seg(image_seg, point_seg)

        # 2. Backbone
        '''
            backbone totally costs 0.012s
        '''
        img_feats_list = self.img_backbone(img_feats)
        img_feats_x = img_feats_list[-1]  # (B, C8, H/8, W/8), aka, (1, 512, 60, 80)
        img_feats_f = img_feats_list[0]   # (B, C2, H, W), aka, (1, 128, 480, 640)
        pcd_feats_list = self.pcd_backbone(pcd_feats, data_dict)
        pcd_feats_c = pcd_feats_list[-1]  # (Nc, 1024)
        pcd_feats_f = pcd_feats_list[0]   # (Nf, 128)
        '''
            obtain feature map of each segmentation primitives with coordinates
        '''
        img_corrds = data_dict["pixel_coords"].detach()
        pts_corrds = data_dict["points"][0].detach()
        img_fea_delt_sels, pts_fea_delt_sels, img_cor_delt_sels, pts_cor_delt_sels, \
        img_fea_mean_sels, pts_fea_mean_sels, img_cor_mean_sels, pts_cor_mean_sels = \
            self.fea_generate_from_mask(
            img_feats_f, pcd_feats_f, img_corrds,   pts_corrds,
            num_img_seg, num_pts_seg, img_seg_mask, pts_seg_mask)

        # discard somethings due to the limite gpu memory
        # data_dict.pop("points")
        data_dict.pop("neighbors")
        data_dict.pop("subsampling")
        data_dict.pop("upsampling")
        # print("backbone cost: ", time.time()-start_time)

        # 3. Interaction 
        '''
            construct a interaction network with segmentation primitives
        '''
        # 3.1 construct a primitives matching matrix for supervision
        img_fea_mean_array = torch.zeros((num_img_seg,128)).cuda()
        pts_fea_mean_array = torch.zeros((num_pts_seg,128)).cuda()
        for i in range(num_img_seg):
            img_fea_mean_array[i,:] = img_fea_mean_sels[i]
            # print(img_fea_mean_sels[i])
        for i in range(num_pts_seg):
            pts_fea_mean_array[i,:] = pts_fea_mean_sels[i]
            # print(pts_fea_mean_sels[i])
        img_fea_mean_array = img_fea_mean_array.unsqueeze(0)
        pts_fea_mean_array = pts_fea_mean_array.unsqueeze(0)
        img_fea_mean_array = torch.nn.functional.normalize(img_fea_mean_array, p=2, dim=-1)
        pts_fea_mean_array = torch.nn.functional.normalize(pts_fea_mean_array, p=2, dim=-1)
        
        # mat_primitive = torch.matmul(img_fea_mean_array[0], pts_fea_mean_array[0].t())
        # output_dict['mat_primitive'] = mat_primitive

        mat_primitive = pairwiseL2Dist(img_fea_mean_array, pts_fea_mean_array)
        # Sinkhorn:
        # Set replicated points to have a zero prior probability
        b, m, n = mat_primitive.size()
        r = mat_primitive.new_zeros((b, m))  # bxm
        c = mat_primitive.new_zeros((b, n))  # bxn
        r[0, : num_img_seg] = 1.0 / num_img_seg
        c[0, : num_pts_seg] = 1.0 / num_pts_seg
        mat_primitive = self.sinkhorn(mat_primitive, r, c)[0]
        mat_primitive = torch.softmax(mat_primitive, dim=1)
        output_dict['mat_primitive'] = mat_primitive

        # print("check: ", torch.sum(mat_primitive))
        # assert 1==-1
        
        # 3.2 feature interaction with primitives matching matrix
        # TODO

        # 3. Transformer
        '''
            transformer totally costs 0.080s 
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
            pyramid totally costs 0.002s
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

            it totally costs 0.064s
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
            all forward procedure totally costs 0.172s
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
    model = fewshotI2P(cfg)
    return model

