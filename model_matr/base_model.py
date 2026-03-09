import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union

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
from .image_backbone import FeaturePyramid, ImageBackbone
from .point_backbone import PointBackbone
from .fusion_module  import CrossModalFusionModule
from .cov_diff_net      import CovDiFF
from .cov_diff_net_full import CovDiFF_full 

from .utils import get_2d3d_node_correspondences, patchify
from vision3d.ops import knn_interpolate

import open3d as o3d
import numpy as np

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

class baseI2P(nn.Module):
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
        
        # --- unused
        self.cov_net_0 = CovDiFF_full()
        self.cov_net_1 = CovDiFF_full()
        self.cov_net_2 = CovDiFF_full()

        self.coarse_target = SuperPointProposalGenerator(
            cfg.model.coarse_matching.num_targets,
            cfg.model.coarse_matching.overlap_threshold)

        self.coarse_matching = SuperPointMatchingMutualTopk(
            cfg.model.coarse_matching.num_correspondences,
            k=cfg.model.coarse_matching.topk,
            threshold=cfg.model.coarse_matching.similarity_threshold)
                 
    def back_project_depth(self,
        depth_mat: Tensor,
        intrinsics: Tensor,
        scaling_factor_a: float = 1000.0,
        scaling_factor_b: float = 1000.0,
        depth_limit: Optional[float] = None,
        transposed: bool = False,
        return_mask: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Back project depth image to point cloud.

        Args:
            depth_mat (Tensor): the depth image in the shape of (B, H, W).
            intrinsics (Tensor): the intrinsic matrix in the shape of (B, 3, 3).
            scaling_factor (float): the depth scaling factor. Default: 1000.
            depth_limit (float, optional): ignore the pixels further than this value.
            transposed (bool): if True, the resulting point matrix is in the shape of (B, H, W, 3).
            return_mask (bool): if True, return a mask matrix where 0-depth points are False. Default: False.

        Returns:
            A Tensor of the point image in the shape of (B, 3, H, W).
            A Tensor of the mask image in the shape of (B, H, W).
        """
        focal_x = intrinsics[..., 0:1, 0:1]
        focal_y = intrinsics[..., 1:2, 1:2]
        center_x = intrinsics[..., 0:1, 2:3]
        center_y = intrinsics[..., 1:2, 2:3]

        batch_size, height, width = depth_mat.shape
        coords = torch.arange(height * width).view(height, width).to(depth_mat.device).unsqueeze(0).expand_as(depth_mat)
        u = coords % width  # (B, H, W)
        v = torch.div(coords, width, rounding_mode="floor")  # (B, H, W)

        z = depth_mat * scaling_factor_a + scaling_factor_b  # (B, H, W)
        if depth_limit is not None:
            z.masked_fill_(torch.gt(z, depth_limit), 0.0)
        x = (u - center_x) * z / focal_x  # (B, H, W)
        y = (v - center_y) * z / focal_y  # (B, H, W)

        if transposed:
            points = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)
        else:
            points = torch.stack([x, y, z], dim=1)  # (B, 3, H, W)

        if not return_mask:
            return points

        masks = torch.gt(z, 0.0)

        return points, masks

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


