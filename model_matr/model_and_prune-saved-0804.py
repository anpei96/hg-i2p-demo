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
import cv2 as cv
from .base_model import baseI2P

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
            scales = scales * 1000

            # histogram can only be used in cpu mode
            max_bins = 1000*100
            hist_scales = scales.cpu().histogram(bins=max_bins,range=(0, torch.max(scales)))
            visual_hist = hist_scales.hist.reshape((-1)).cuda()
            visual_bins = hist_scales.bin_edges.reshape((-1)).cuda()

            # scale estimated by the peak of histogram
            idx_max = torch.argmax(visual_hist)
            sca_max = visual_bins[idx_max]
            max_res = 500
            scale_a = sca_max - torch.max(scales)/max_bins * max_res
            scale_b = sca_max + torch.max(scales)/max_bins * max_res
            valid_mask = (scale_colls >= scale_a/1000) \
                & (scale_colls <= scale_b/1000)
            scale_colls[~valid_mask] = 0 
            scale_colls[valid_mask]  = 1
            # print("sca_max: ", sca_max)
            print("===> scale region: ", scale_a, scale_b)

            map_scale = torch.sum(scale_colls, dim=1)
            valid_mask = (map_scale > 0)#torch.mean(map_scale)) # torch.mean(map_scale)
            valid_mask = valid_mask.reshape((-1))

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
            # output_dict["img_corr_points"] = \
            #     output_dict["img_corr_points"][valid_mask]
            # output_dict["img_corr_pixels"] = \
            #     output_dict["img_corr_pixels"][valid_mask]
            # output_dict["img_corr_indices"] = \
            #     output_dict["img_corr_indices"][valid_mask]
            # output_dict["pcd_corr_points"] = \
            #     output_dict["pcd_corr_points"][valid_mask]
            # output_dict["pcd_corr_pixels"] = \
            #     output_dict["pcd_corr_pixels"][valid_mask]
            # output_dict["pcd_corr_indices"] = \
            #     output_dict["pcd_corr_indices"][valid_mask]
            # output_dict["corr_scores"] = \
            #     output_dict["corr_scores"][valid_mask]

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

