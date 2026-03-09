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
            corr_2d_a = torch.ones((num_cor,3)).cuda()
            corr_2d_a[:,0] = corr_2d[:,1]
            corr_2d_a[:,1] = corr_2d[:,0]
            temp_ww = torch.matmul(corr_2d_a, intr_inv_t) # [d,3]
            temp_w1 = temp_ww.repeat((num_cor,1,1))       # [d,d,3]
            temp_w2 = temp_w1.transpose(1,0)              # [d,d,3]

            d_vec = depth_vec[corr_2d_ind].reshape((-1,1)) # [d,1]
            temp_d1 = d_vec.repeat((num_cor,1,1))          # [d,d,1]
            temp_d2 = temp_d1.transpose(1,0)               # [d,d,1]
            d_coll  = temp_d2/temp_d1                      # [d,d,1]
            d_colls = \
                torch.concat((d_coll,d_coll,d_coll),dim=2)  # [d,d,3]
            w_colls = temp_w1 - torch.mul(d_colls, temp_w2) # [d,d,3]

            '''
                step 2 scale-rotation-inliers optimization
            '''
            # step 2.1 scale estimation
            #   also this sub-step is used for check w and v correctness
            v_colls_norm = \
                torch.norm(v_colls, dim=2, keepdim=True) # [d,d,1]
            w_colls_norm = \
                torch.norm(w_colls, dim=2, keepdim=True) # [d,d,1]
            v_colls_unit = \
                torch.nn.functional.normalize(v_colls, dim=2) # [d,d,3]
            w_colls_unit = \
                torch.nn.functional.normalize(w_colls, dim=2) # [d,d,3]

            # coarse scale estimation            
            scale_colls = (v_colls_norm/w_colls_norm)/temp_d1 # [d,d,1]
            scale_colls = \
                torch.where(torch.isnan(scale_colls), torch.full_like(scale_colls, 0), scale_colls)
            scale_colls = \
                torch.where(torch.isinf(scale_colls), torch.full_like(scale_colls, 0), scale_colls)

            # l2 distance mask (largely eliminate unimportant candidates)
            distance_mask = \
                (torch.norm(v_colls, dim=2, keepdim=True) <=3.25) # [d,d,1]
            distance_mask = distance_mask.float()
            tmp = torch.diag(distance_mask[:,:,0])
            tmp = torch.diag(tmp, 0)
            distance_mask[:,:,0] = distance_mask[:,:,0] - tmp
            scale_colls_f = torch.mul(scale_colls, distance_mask)

            # multi-scale estimation via histgorm
            valid_mask = (scale_colls_f != 0)
            scales = scale_colls_f[valid_mask] # [N] from [d,d,1] => [N]
            scales = \
                torch.where(torch.isnan(scales), torch.full_like(scales, 0), scales)
            scales = \
                torch.where(torch.isinf(scales), torch.full_like(scales, 0), scales)

            # histogram can only be used in cpu mode
            max_bins = 1024*8
            hist_scales = scales.cpu().histogram(bins=max_bins,range=(0, torch.max(scales)))
            visual_hist = hist_scales.hist.reshape((-1)).cuda()
            visual_bins = hist_scales.bin_edges.reshape((-1)).cuda()

            # scale estimated by the peak of histogram
            idx_max = torch.argmax(visual_hist)
            sca_max = visual_bins[idx_max]
            max_res = 1
            scale_a = sca_max #- torch.max(scales)/max_bins * max_res
            scale_b = sca_max + torch.max(scales)/max_bins * max_res
            valid_mask = (scale_colls >= scale_a) & (scale_colls <= scale_b)
            scale_colls[~valid_mask] = 0 
            scale_colls[valid_mask]  = 1

            map_scale = torch.sum(scale_colls, dim=1)
            valid_mask = (map_scale > torch.mean(map_scale)*0.25) # torch.mean(map_scale)
            valid_mask = valid_mask.reshape((-1))

            # === visualize match matrix --- ok === #
            # import cv2 as cv
            # import numpy as np
            # a = distance_mask[:,:,0].detach().cpu().numpy()
            # a = (a*255).astype(np.uint8)
            # a = cv.resize(a, (640,640))
            # cv.imshow("a", a)
            # cv.waitKey(0)

            # === visualize histogram --- ok === #
            import matplotlib.pyplot as plt
            visual_hist = hist_scales.hist.detach().numpy().reshape((-1))
            visual_bins = hist_scales.bin_edges.detach().numpy().reshape((-1))
            vis_bins = 1024
            plt.plot(visual_bins[:vis_bins], visual_hist[:vis_bins])
            plt.show()

            # === visualize histogram for the gt scale --- ok === #
            # ratio_gt = data_dict["ratio_mat"].detach().reshape((-1))
            # valid_mask_ = (ratio_gt > 0)
            # ratio_gt = ratio_gt[valid_mask_]
            # print(ratio_gt)
            # print(torch.max(ratio_gt))
            # print(torch.mean(ratio_gt))
            # ratio_gt = \
            #     torch.where(torch.isnan(ratio_gt), torch.full_like(ratio_gt, 0), ratio_gt)
            # ratio_gt = \
            #     torch.where(torch.isinf(ratio_gt), torch.full_like(ratio_gt, 0), ratio_gt)
            # hist_scales_ = ratio_gt.cpu().histogram(bins=max_bins,range=(0, torch.max(scales)))
            # visual_hist_ = hist_scales_.hist.detach().numpy().reshape((-1))
            # visual_bins_ = hist_scales_.bin_edges.detach().numpy().reshape((-1))
            # plt.plot(visual_bins_[:vis_bins], visual_hist_[:vis_bins])
            # plt.show()

            # for debug usage
            # transform = data_dict["transform"].detach()
            # print("scale_colls: ", scale_colls.size())
            # print("scales: ", scales.size(), torch.min(scales), torch.max(scales))
            print("sca_bax: ", sca_max, " with numbers: ", visual_hist[idx_max])
            print("==> final corr num: ", torch.sum(valid_mask))
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

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration
        return output_dict

def create_model(cfg):
    model = baseline_with_prune(cfg)
    return model

