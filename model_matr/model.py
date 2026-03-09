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
    render,
)

# isort: split
from .fusion_module  import CrossModalFusionModule
from .image_backbone import FeaturePyramid, ImageBackbone
from .point_backbone import PointBackbone
from .cov_diff_net      import CovDiFF
from .cov_diff_net_full import CovDiFF_full 

from .utils import get_2d3d_node_correspondences, patchify
from vision3d.ops import knn_interpolate_pack_mode

class baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
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

        self.coarse_target = SuperPointProposalGenerator(
            cfg.model.coarse_matching.num_targets,
            cfg.model.coarse_matching.overlap_threshold)

        self.coarse_matching = SuperPointMatchingMutualTopk(
            cfg.model.coarse_matching.num_correspondences,
            k=cfg.model.coarse_matching.topk,
            threshold=cfg.model.coarse_matching.similarity_threshold)

        # self.cov_net_0 = CovDiFF()
        # self.cov_net_1 = CovDiFF()
        # self.cov_net_2 = CovDiFF()
        # ---- d
        self.cov_net_0 = CovDiFF_full()
        self.cov_net_1 = CovDiFF_full()
        self.cov_net_2 = CovDiFF_full()

    def genenrate_label(self, output_dict):
         # 4.1 Generate 3d patches
        _, pcd_node_sizes, pcd_node_masks, pcd_node_knn_indices, pcd_node_knn_masks = point_to_node_partition(
            output_dict["pcd_points_f"],
            output_dict["pcd_points_c"],
            self.pcd_num_points_in_patch,
            gather_points=True,
            return_count=True,
        )
        output_dict["pcd_node_knn_indices"] = pcd_node_knn_indices
        output_dict["pcd_node_knn_masks"]   = pcd_node_knn_masks

        pcd_node_masks = torch.logical_and(pcd_node_masks, torch.gt(pcd_node_sizes, self.pcd_min_node_size))
        pcd_padded_points_f = torch.cat([output_dict["pcd_points_f"], torch.ones_like(output_dict["pcd_points_f"][:1]) * 1e10], dim=0)
        pcd_node_knn_points = index_select(pcd_padded_points_f, pcd_node_knn_indices, dim=0)
        pcd_padded_pixels_f = torch.cat([output_dict["pcd_pixels_f"], torch.ones_like(output_dict["pcd_pixels_f"][:1]) * 1e10], dim=0)
        pcd_node_knn_pixels = index_select(pcd_padded_pixels_f, pcd_node_knn_indices, dim=0)
        output_dict["pcd_node_masks"] = pcd_node_masks

        # 4.2 Generate 2d patches
        all_img_node_knn_points = []
        all_img_node_knn_pixels = []
        all_img_node_knn_indices = []
        all_img_node_knn_masks = []
        all_img_node_masks = []
        all_img_node_levels = []
        all_img_num_nodes = []
        all_img_total_nodes = []
        total_img_num_nodes = 0

        all_gt_img_node_corr_levels = []
        all_gt_img_node_corr_indices = []
        all_gt_pcd_node_corr_indices = []
        all_gt_img_node_corr_overlaps = []
        all_gt_pcd_node_corr_overlaps = []

        img_h_c = self.img_h_c
        img_w_c = self.img_w_c
        for i in range(self.img_num_levels_c):
            (
                img_node_knn_points,  # (N, Ki, 3)
                img_node_knn_pixels,  # (N, Ki, 2)
                img_node_knn_indices,  # (N, Ki)
                img_node_knn_masks,  # (N, Ki)
                img_node_masks,  # (N)
            ) = patchify(
                output_dict["img_points_f"],
                output_dict["img_pixels_f"],
                output_dict["img_masks_f"],
                output_dict["img_h_f"],
                output_dict["img_w_f"],
                img_h_c,
                img_w_c,
                stride=2,
            )

            img_num_nodes = img_h_c * img_w_c
            img_node_levels = torch.full(size=(img_num_nodes,), fill_value=i, dtype=torch.long).cuda()

            all_img_node_knn_points.append(img_node_knn_points)
            all_img_node_knn_pixels.append(img_node_knn_pixels)
            all_img_node_knn_indices.append(img_node_knn_indices)
            all_img_node_knn_masks.append(img_node_knn_masks)
            all_img_node_masks.append(img_node_masks)
            all_img_node_levels.append(img_node_levels)
            all_img_num_nodes.append(img_num_nodes)
            all_img_total_nodes.append(total_img_num_nodes)

            output_dict["all_img_node_knn_points"]  = all_img_node_knn_points
            output_dict["all_img_node_knn_pixels"]  = all_img_node_knn_pixels
            output_dict["all_img_node_knn_indices"] = all_img_node_knn_indices
            output_dict["all_img_total_nodes"] = all_img_total_nodes

            # print("img_node_knn_points: ", img_node_knn_points.size())
            # print("pcd_node_knn_points: ", pcd_node_knn_points.size())

            # 4.3 Generate coarse-level ground truth
            (
                gt_img_node_corr_indices,
                gt_pcd_node_corr_indices,
                gt_img_node_corr_overlaps,
                gt_pcd_node_corr_overlaps,
            ) = get_2d3d_node_correspondences(
                img_node_masks,
                img_node_knn_points,
                img_node_knn_pixels,
                img_node_knn_masks,
                pcd_node_masks,
                pcd_node_knn_points,
                pcd_node_knn_pixels,
                pcd_node_knn_masks,
                output_dict["transform"],
                self.matching_radius_2d,
                self.matching_radius_3d,
            )

            gt_img_node_corr_indices += total_img_num_nodes
            gt_img_node_corr_levels = torch.full_like(gt_img_node_corr_indices, fill_value=i)
            all_gt_img_node_corr_levels.append(gt_img_node_corr_levels)
            all_gt_img_node_corr_indices.append(gt_img_node_corr_indices)
            all_gt_pcd_node_corr_indices.append(gt_pcd_node_corr_indices)
            all_gt_img_node_corr_overlaps.append(gt_img_node_corr_overlaps)
            all_gt_pcd_node_corr_overlaps.append(gt_pcd_node_corr_overlaps)

            img_h_c //= 2
            img_w_c //= 2
            total_img_num_nodes += img_num_nodes

        img_node_masks = torch.cat(all_img_node_masks, dim=0)
        img_node_levels = torch.cat(all_img_node_levels, dim=0)

        output_dict["img_num_nodes"] = total_img_num_nodes
        output_dict["pcd_num_nodes"] = output_dict["pcd_points_c"].shape[0]
        output_dict["img_node_masks"]  = img_node_masks
        output_dict["img_node_levels"] = img_node_levels

        gt_img_node_corr_levels = torch.cat(all_gt_img_node_corr_levels, dim=0)
        gt_img_node_corr_indices = torch.cat(all_gt_img_node_corr_indices, dim=0)
        gt_pcd_node_corr_indices = torch.cat(all_gt_pcd_node_corr_indices, dim=0)
        gt_img_node_corr_overlaps = torch.cat(all_gt_img_node_corr_overlaps, dim=0)
        gt_pcd_node_corr_overlaps = torch.cat(all_gt_pcd_node_corr_overlaps, dim=0)

        gt_node_corr_min_overlaps = torch.minimum(gt_img_node_corr_overlaps, gt_pcd_node_corr_overlaps)
        gt_node_corr_max_overlaps = torch.maximum(gt_img_node_corr_overlaps, gt_pcd_node_corr_overlaps)

        output_dict["gt_img_node_corr_indices"] = gt_img_node_corr_indices
        output_dict["gt_pcd_node_corr_indices"] = gt_pcd_node_corr_indices
        output_dict["gt_img_node_corr_overlaps"] = gt_img_node_corr_overlaps
        output_dict["gt_pcd_node_corr_overlaps"] = gt_pcd_node_corr_overlaps
        output_dict["gt_img_node_corr_levels"] = gt_img_node_corr_levels
        output_dict["gt_node_corr_min_overlaps"] = gt_node_corr_min_overlaps
        output_dict["gt_node_corr_max_overlaps"] = gt_node_corr_max_overlaps
        
        return output_dict

    def unpack_2d_3d_data(self, data_dict, output_dict):
        '''
            a little change if the input is normal
            [B,480,640,3] => [B,1,480,640,3] => [B,3,480,640]
        '''
        # 2d image branch
        image = data_dict["image"].unsqueeze(1).detach() 
        image = image.transpose(1, -1)
        image = image.squeeze(-1)

        depth = data_dict["depth"].detach()  # (B, H, W)
        intrinsics = data_dict["intrinsics"].detach()  # (B, 3, 3)
        transform = data_dict["transform"].detach()

        img_h = image.shape[2]
        img_w = image.shape[3]
        img_h_f = img_h
        img_w_f = img_w
        output_dict["transform"] = transform
        output_dict["img_h_f"] = img_h_f
        output_dict["img_w_f"] = img_w_f

        # use normalized pixel coordinates for transformer
        img_pixels_c = create_meshgrid(self.img_h_c, self.img_w_c, normalized=True, flatten=True)  # (768, 2)
        output_dict["img_pixels_c"] = img_pixels_c

        img_points, img_masks = back_project(depth, intrinsics, depth_limit=6.0, transposed=True, return_mask=True)
        img_points = img_points.squeeze(0)  # (B, H, W, 3) -> (H, W, 3)
        img_masks = img_masks.squeeze(0)  # (B, H, W) -> (H, W)
        img_pixels = create_meshgrid(img_h, img_w).float()  # (H, W, 2)

        img_points_f = img_points  # (H, H, 3)
        img_masks_f = img_masks  # (H, H)
        img_pixels_f = img_pixels  # (H, W, 2)

        img_points = img_points.view(-1, 3)  # (H, W, 3) -> (HxW, 3)
        img_pixels = img_pixels.view(-1, 2)  # (H, W, 2) -> (HxW, 2)
        img_masks  = img_masks.view(-1)  # (H, W) -> (HxW)
        img_points_f = img_points_f.view(-1, 3)  # (H, W, 3) -> (HxW, 3)
        img_pixels_f = img_pixels_f.view(-1, 2)  # (H/2xW/2, 2)
        img_masks_f  = img_masks_f.view(-1)  # (H, W) -> (HxW)

        output_dict["img_points"] = img_points
        output_dict["img_pixels"] = img_pixels
        output_dict["img_masks"] = img_masks
        output_dict["img_points_f"] = img_points_f
        output_dict["img_pixels_f"] = img_pixels_f
        output_dict["img_masks_f"] = img_masks_f

        # 3d point cloud branch
        # pcd_feats  = data_dict["feats"].detach()
        pcd_points = data_dict["points"][0].detach()
        pcd_points_f = data_dict["points"][0].detach()
        pcd_points_c = data_dict["points"][-1].detach()
        pcd_pixels_f = render(pcd_points_f, intrinsics, extrinsics=transform, rounding=False)

        output_dict["pcd_points"] = pcd_points
        output_dict["pcd_points_c"] = pcd_points_c
        output_dict["pcd_points_f"] = pcd_points_f
        output_dict["pcd_pixels_f"] = pcd_pixels_f

        return image, output_dict

    def post_process_generate_corres(self, img_feats_c, pcd_feats_c,
        img_feats_f, pcd_feats_f, output_dict):
        (
            img_node_corr_indices,
            pcd_node_corr_indices,
            node_corr_scores,
        ) = self.coarse_matching(img_feats_c, pcd_feats_c, output_dict["img_node_masks"], output_dict["pcd_node_masks"])
        img_node_corr_levels = output_dict["img_node_levels"][img_node_corr_indices]

        output_dict["img_node_corr_indices"] = img_node_corr_indices
        output_dict["pcd_node_corr_indices"] = pcd_node_corr_indices
        output_dict["img_node_corr_levels"] = img_node_corr_levels

        pcd_padded_feats_f = torch.cat([pcd_feats_f, torch.zeros_like(pcd_feats_f[:1])], dim=0)

        # 7. Extract patch correspondences
        all_img_corr_indices = []
        all_pcd_corr_indices = []

        for i in range(self.img_num_levels_c):
            node_corr_masks = torch.eq(img_node_corr_levels, i)

            if node_corr_masks.sum().item() == 0:
                continue

            cur_img_node_corr_indices = img_node_corr_indices[node_corr_masks] - output_dict["all_img_total_nodes"][i]
            cur_pcd_node_corr_indices = pcd_node_corr_indices[node_corr_masks]

            img_node_knn_points  = output_dict["all_img_node_knn_points"][i]
            img_node_knn_pixels  = output_dict["all_img_node_knn_pixels"][i]
            img_node_knn_indices = output_dict["all_img_node_knn_indices"][i]

            img_node_corr_knn_indices = index_select(img_node_knn_indices, cur_img_node_corr_indices, dim=0)
            img_node_corr_knn_masks = torch.ones_like(img_node_corr_knn_indices, dtype=torch.bool)
            img_node_corr_knn_feats = index_select(img_feats_f, img_node_corr_knn_indices, dim=0)

            pcd_node_corr_knn_indices = output_dict["pcd_node_knn_indices"][cur_pcd_node_corr_indices]  # (P, Kc)
            pcd_node_corr_knn_masks = output_dict["pcd_node_knn_masks"][cur_pcd_node_corr_indices]  # (P, Kc)
            pcd_node_corr_knn_feats = index_select(pcd_padded_feats_f, pcd_node_corr_knn_indices, dim=0)

            similarity_mat = pairwise_cosine_similarity(
                img_node_corr_knn_feats, pcd_node_corr_knn_feats, normalized=True
            )

            batch_indices, row_indices, col_indices, _ = batch_mutual_topk_select(
                similarity_mat,
                k=1,
                row_masks=img_node_corr_knn_masks,
                col_masks=pcd_node_corr_knn_masks,
                threshold=0.75,
                largest=True,
                mutual=True,
            )

            img_corr_indices = img_node_corr_knn_indices[batch_indices, row_indices]
            pcd_corr_indices = pcd_node_corr_knn_indices[batch_indices, col_indices]

            all_img_corr_indices.append(img_corr_indices)
            all_pcd_corr_indices.append(pcd_corr_indices)

        img_corr_indices = torch.cat(all_img_corr_indices, dim=0)
        pcd_corr_indices = torch.cat(all_pcd_corr_indices, dim=0)

        # duplicate removal
        num_points_f = output_dict["pcd_points_f"].shape[0]
        corr_indices = img_corr_indices * num_points_f + pcd_corr_indices
        unique_corr_indices = torch.unique(corr_indices)
        img_corr_indices = torch.div(unique_corr_indices, num_points_f, rounding_mode="floor")
        pcd_corr_indices = unique_corr_indices % num_points_f

        img_points_f = output_dict["img_points_f"].view(-1, 3)
        img_pixels_f = output_dict["img_pixels_f"].view(-1, 2)
        img_corr_points = img_points_f[img_corr_indices]
        img_corr_pixels = img_pixels_f[img_corr_indices]
        pcd_corr_points = output_dict["pcd_points_f"][pcd_corr_indices]
        pcd_corr_pixels = output_dict["pcd_points_f"][pcd_corr_indices]
        img_corr_feats = img_feats_f[img_corr_indices]
        pcd_corr_feats = pcd_feats_f[pcd_corr_indices]
        corr_scores = (img_corr_feats * pcd_corr_feats).sum(1)

        output_dict["img_corr_points"] = img_corr_points
        output_dict["img_corr_pixels"] = img_corr_pixels
        output_dict["img_corr_indices"] = img_corr_indices
        output_dict["pcd_corr_points"] = pcd_corr_points
        output_dict["pcd_corr_pixels"] = pcd_corr_pixels
        output_dict["pcd_corr_indices"] = pcd_corr_indices
        output_dict["corr_scores"] = corr_scores
        return output_dict

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

        '''
            note-0524
            feature interaction layer --- cov-net
            in the fine-tuning stage, open the above code to train them
        '''
        # img_feats_f, pcd_feats_f = \
        #     self.cov_net_0(data_dict, img_feats_f, pcd_feats_f)
        # img_feats_f, pcd_feats_f = \
        #     self.cov_net_1(data_dict, img_feats_f, pcd_feats_f)
        # img_feats_f, pcd_feats_f = \
        #     self.cov_net_2(data_dict, img_feats_f, pcd_feats_f)

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

        torch.cuda.synchronize()
        duration = time.time() - start_time
        output_dict["duration"] = duration
        return output_dict

def create_model(cfg):
    model = baseline(cfg)
    return model

