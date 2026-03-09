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
            
            import cv2 as cv
            ct_3d = corr_3d.detach().cpu().numpy()
            ct_2d = corr_2d.detach().cpu().numpy()
            k_mat = intrinsics.detach().cpu().numpy()
            ct_2d_ = ct_2d * 1.0
            ct_2d_[:,0] = ct_2d[:,1]
            ct_2d_[:,1] = ct_2d[:,0]
            success, rvec, tvec, inliers = \
                cv.solvePnPRansac(ct_3d, ct_2d_, k_mat, None,
                iterationsCount=5000, reprojectionError=5, flags=cv.SOLVEPNP_P3P)
            valid_mask = inliers.reshape((-1))
            # print("inliers: ", inliers.shape)
            # print(inliers)
            # print("rvec: ", rvec)
            # print("tvec: ", tvec)
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
            print("===> img_corr_points: ", output_dict["img_corr_points"].size())

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration
        return output_dict

def create_model(cfg):
    model = baseline_with_prune(cfg)
    return model

