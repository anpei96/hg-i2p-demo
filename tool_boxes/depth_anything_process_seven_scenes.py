import os
import glob
import numpy as np
import torch
import cv2 as cv
import time as time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union

from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

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

def back_project_depth(
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

def main_batch_process():
    '''
        a standard batch process to predict depth from the seven scenes dataset
    '''
    transform = Compose([
        Resize(
            width=630,#self.img_w_c,
            height=476,#self.img_h_c,                   
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_path = '/media/anpei/DiskA/05_i2p_fewshot/tool_boxes/depth_anything/checkpoints/'
    cfg_path = checkpoint_path + 'config.json'
    pth_path = checkpoint_path + 'depth_anything_vits14.pth'

    with open(cfg_path) as f:
        cfg = json.load(f)
    weights = torch.load(pth_path)
    depth_model = DepthAnything(cfg).to(DEVICE).eval()
    depth_model.load_state_dict(weights)
    depth_coffa = torch.tensor([1.0], requires_grad=True).to(DEVICE)
    depth_coffb = torch.tensor([0.0], requires_grad=True).to(DEVICE)

    '''
    note-0516
        anpei dataset path 
    '''
    data_dir = "/media/anpei/DiskA/05_i2p_fewshot/data/"
    # data_base_path = data_dir+"7Scenes/data/chess"
    # path_num = 6
    # data_base_path = data_dir+"7Scenes/data/fire"
    # path_num = 4
    # data_base_path = data_dir+"7Scenes/data/heads"
    # path_num = 2
    # data_base_path = data_dir+"7Scenes/data/office"
    # path_num = 10
    # data_base_path = data_dir+"7Scenes/data/pumpkin"
    # path_num = 8
    # data_base_path = data_dir+"7Scenes/data/redkitchen"
    # path_num = 14
    data_base_path = data_dir+"7Scenes/data/stairs"
    path_num = 6

    for idx in range(path_num):
        idx = idx + 1
        spec_path = data_base_path + "/seq-" + str("%02d" % idx) + "/"
        img_paths = glob.glob(spec_path + "color_*.png")
        img_paths.sort()

        with torch.no_grad():
            for img_path in img_paths:
                '''
                    remove dsine/sam/depth preprocessed image
                '''
                if img_path[-9:-4] == "dsine":
                    continue
                if img_path[-9:-4] == "sampo":
                    continue
                if img_path[-9:-4] == "sampd":
                    continue
                if img_path[-9:-4] == "depth":
                    continue
                print(img_path)

                time_st = time.time()
                image = cv.imread(img_path)
                image = image/255.0
                image_for_depth = transform({'image': image})['image']    
                image_for_depth = torch.from_numpy(image_for_depth).unsqueeze(0).cuda()
                image_depth_any = depth_model(image_for_depth)
                image_depth_any = 36-image_depth_any # 36 is a megic number :)
                time_ed = time.time()
                print("cost time: ", time_ed-time_st)

                '''
                    visulize predicted depth
                '''
                import numpy as np
                depth = image_depth_any[0]
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.cpu().numpy().astype(np.uint8)
                depth = cv.applyColorMap(depth, cv.COLORMAP_INFERNO)
                # cv.imshow("depth", depth)
                # cv.waitKey(0)

                '''
                    save predicted results
                '''
                target_path = img_path[:-4] + "_depth.png"
                cv.imwrite(target_path, depth)

                '''
                    visualize point cloud (affine transform) from predicted depth
                '''
                is_need_visualize = False
                if is_need_visualize:
                    fx = 5.85000000e+02
                    fy = 5.85000000e+02
                    cx = 3.20000000e+02
                    cy = 2.40000000e+02
                    intrinsics = torch.tensor([
                        [fx,  0, cx],
                        [ 0, fy, cy],
                        [ 0,  0,  1]
                    ], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    img_points_da, img_masks_da = back_project_depth(
                        image_depth_any, intrinsics, depth_limit=64.0, 
                        scaling_factor_a=depth_coffa, 
                        scaling_factor_b=depth_coffb, 
                        transposed=True, return_mask=True)
                    
                    pts_vis = img_points_da.detach().cpu().numpy().reshape((-1,3))
                    print("pts_vis: ", pts_vis.shape)
                    pcd_vis = o3d.geometry.PointCloud()
                    pcd_vis.points = o3d.utility.Vector3dVector(pts_vis)
                    show_pcd(pcd_vis)

if __name__ == '__main__':
    '''
        process a whole dataset
    '''
    main_batch_process()