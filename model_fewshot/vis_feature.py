from this import d
import time
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import numpy  as np

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

def PCA(X, k=3):
    # X_mean = torch.mean(X, 0)
    # X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.matmul(X.T, X))
    return torch.matmul(X, U[:,:k]), U[:,:k]

def latent_feature_vis(
    img_feats_f_bef, pcd_feats_f_bef,
    img_feats_f_aft, pcd_feats_f_aft):
    '''
    input features:
        img_feats_f_bef:  torch.Size([1, 128, 480, 640])
        pcd_feats_f_bef:  torch.Size([30000, 128])
        img_feats_f_aft:  torch.Size([1, 128, 480, 640])
        pcd_feats_f_aft:  torch.Size([30000, 128])
    '''
    fi_bef = img_feats_f_bef.reshape((128,-1)).T
    fp_bef = pcd_feats_f_bef
    fi_aft = img_feats_f_aft.reshape((128,-1)).T
    fp_aft = pcd_feats_f_aft

    fi_bef = fi_bef[::10,] # 10x down-sample
    fi_aft = fi_aft[::10,]

    print("fi_bef: ", fi_bef)
    print("fi_aft: ", fi_aft)

    # first project then into a 3d space
    # xyz_ii_b = fi_bef[:,6:9]
    # xyz_pp_b = fp_bef[:,6:9]
    # xyz_ii_a = fi_aft[:,6:9]
    # xyz_pp_a = fp_aft[:,6:9]

    # xyz_ii_b = fi_bef[:,0:3]
    # xyz_pp_b = fp_bef[:,0:3]
    # xyz_ii_a = fi_aft[:,0:3]
    # xyz_pp_a = fp_aft[:,0:3]
    
    # xyz_ii_b = torch.pca_lowrank(fi_bef, 3)[0]
    # xyz_pp_b = torch.pca_lowrank(fp_bef, 3)[0]
    # xyz_ii_a = torch.pca_lowrank(fi_aft, 3)[0]
    # xyz_pp_a = torch.pca_lowrank(fp_aft, 3)[0]

    # xyz_ii_b = PCA(fi_bef, 3)
    # xyz_pp_b = PCA(fp_bef, 3)
    # xyz_ii_a = PCA(fi_aft, 3)
    # xyz_pp_a = PCA(fp_aft, 3)

    xyz_ii_b, proj_mat = PCA(fi_bef, 3)
    xyz_pp_b = torch.matmul(fp_bef, proj_mat)
    xyz_ii_a, proj_mat = PCA(fi_aft, 3)
    xyz_pp_a = torch.matmul(fp_aft, proj_mat)

    # then project them in a 3d sphere
    # xyz_ii_b = xyz_ii_b/torch.norm(xyz_ii_b, dim=1, keepdim=True)
    # xyz_pp_b = xyz_pp_b/torch.norm(xyz_pp_b, dim=1, keepdim=True)
    # xyz_ii_a = xyz_ii_a/torch.norm(xyz_ii_a, dim=1, keepdim=True)
    # xyz_pp_a = xyz_pp_a/torch.norm(xyz_pp_a, dim=1, keepdim=True)

    # transfer them as cpu numpy
    xyz_ii_b = xyz_ii_b.cpu().numpy()
    xyz_pp_b = xyz_pp_b.cpu().numpy()
    xyz_ii_a = xyz_ii_a.cpu().numpy()
    xyz_pp_a = xyz_pp_a.cpu().numpy()

    # visualize with point cloud
    pcd_ii_b = o3d.geometry.PointCloud()
    pcd_ii_b.points = o3d.utility.Vector3dVector(xyz_ii_b[:,:3])
    cc = np.zeros_like(xyz_ii_b[:,:3])
    cc[:,0] = 1.0
    pcd_ii_b.colors = o3d.utility.Vector3dVector(cc[:,:3])

    pcd_pp_b = o3d.geometry.PointCloud()
    pcd_pp_b.points = o3d.utility.Vector3dVector(xyz_pp_b[:,:3])
    cc = np.zeros_like(xyz_pp_b[:,:3])
    cc[:,1] = 1.0
    pcd_pp_b.colors = o3d.utility.Vector3dVector(cc[:,:3])

    show_pcd(pcd_ii_b+pcd_pp_b)

    # ======================================================= #

    pcd_ii_a = o3d.geometry.PointCloud()
    pcd_ii_a.points = o3d.utility.Vector3dVector(xyz_ii_a[:,:3])
    cc = np.zeros_like(xyz_ii_b[:,:3])
    cc[:,0] = 1.0
    pcd_ii_a.colors = o3d.utility.Vector3dVector(cc[:,:3])

    pcd_pp_a = o3d.geometry.PointCloud()
    pcd_pp_a.points = o3d.utility.Vector3dVector(xyz_pp_a[:,:3])
    cc = np.zeros_like(xyz_pp_b[:,:3])
    cc[:,1] = 1.0
    pcd_pp_a.colors = o3d.utility.Vector3dVector(cc[:,:3])

    show_pcd(pcd_ii_a+pcd_pp_a)

    print("xyz_ii_b: ", xyz_ii_b.shape)
    print("xyz_pp_b: ", xyz_pp_b.shape)
    print("xyz_ii_a: ", xyz_ii_a.shape)
    print("xyz_pp_a: ", xyz_pp_a.shape)

    assert 1==-1




