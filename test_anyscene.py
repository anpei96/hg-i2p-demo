import os.path as osp
import time
import numpy as np

'''
    please revise make_cfg for training in other scenes
'''
# from config.seven_scenes_config import make_cfg
from config.seven_scenes_config_ours import make_cfg
# from config.seven_scenes_config_prune import make_cfg
# from config.seven_scenes_config_aug import make_cfg
data_base_path = "/media/anpei/DiskA/05_i2p_fewshot/data/7Scenes/data/"
data_save_path = "/media/anpei/DiskA/05_i2p_fewshot/results/7Scenes/"

# from config.rgbd_scenes_config import make_cfg
# # from config.rgbd_scenes_config_aug import make_cfg
# data_base_path = "/media/anpei/DiskA/05_i2p_fewshot/data/RGBDScenesV2/data/"
# data_save_path = "/media/anpei/DiskA/05_i2p_fewshot/results/RGBDScenesV2/"

# from config.scannet_scenes_config import make_cfg
# # from config.scannet_scenes_config_aug import make_cfg
# data_base_path = "/media/anpei/DiskA/05_i2p_fewshot/data/Scannet/data/"
# data_save_path = "/media/anpei/DiskA/05_i2p_fewshot/results/Scannet/"

# from config.self_v1_scenes_config import make_cfg
# from config.self_v1_scenes_config_aug import make_cfg
# data_base_path = "/media/anpei/DiskA/05_i2p_fewshot/data/Fangwei/data/"
# data_save_path = "/media/anpei/DiskA/05_i2p_fewshot/results/Fangwei/"

# from config.kitti_scenes_config import make_cfg
# data_base_path = ""
# data_save_path = "/media/anpei/DiskA/05_i2p_fewshot/results/Kitti/"
'''
'''
# from dataset.dataset_anyscene import test_data_loader
from dataset.dataset import test_data_loader

# from dataset.dataset_kitti import test_data_loader # only used for kitti
from loss.loss import EvalFunction
'''
    please choose different i2p registration method
'''
# from model_matr.model import create_model
from model_matr.model_and_prune import create_model
# from model_matr.model_and_prune_ransac import create_model
# from model_fewshot.model import create_model
# from model_fewshot.model_plus import create_model
# from model_matr_aug.model import create_model
'''
'''

from vision3d.engine import SingleTester
from vision3d.utils.io import ensure_dir
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import add_tester_args
from vision3d.utils.tensor import tensor_to_array


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        self.log(f"Data loader created: {loading_time:.3f}s collapsed.", level="DEBUG")
        self.log(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(data_loader)
    
        '''
        note-0229:
            our 2d-3d registration model *-*
        '''
        model = create_model(cfg).cuda()
        self.register_model(model)

        '''
        note-0910:
            load other model in test
        '''
        base_path = "/media/anpei/DiskA/05_i2p_fewshot/model_zoos/"
        # ckpt_path = base_path + "top-i2p-kitchen.pth"
        # ckpt_path = base_path + "top-i2p-kitchen-25-1.pth" # no-int
        # ckpt_path = base_path + "baseline-kitchen-real.pth"
        # ckpt_path = base_path + "epoch-5-chess.pth" 
        # ckpt_path = base_path + "baseline-office.pth" 
        # base_path = "/media/anpei/DiskA/05_i2p_fewshot/model_zoo_aug/"
        # ckpt_path = base_path + "kitchen-26.pth"
        # ckpt_path = base_path + "kitchen-20-ablation.pth"
        base_path = "/media/anpei/DiskA/05_i2p_fewshot/model_zoos_plus/"
        # ckpt_path = base_path + "base-office-20.pth"
        # ckpt_path = base_path + "tum-40.pth"
        ckpt_path = base_path + "base-kitchen-30.pth"
        # ckpt_path = base_path + "our-30-tr-align-act-tf-office.pth"
        # ckpt_path = base_path + "our-40-tr-align-act-tf-chess.pth"
        # ckpt_path = base_path + "our-50-tr-align-act-tf-kitchen.pth"
        self.load(ckpt_path)

        # evaluator
        self.eval_func = EvalFunction(cfg).cuda()

        # preparation
        self.output_dir = cfg.exp.cache_dir

        # anpei save index number
        self.save_idx = 0

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.eval_func(data_dict, output_dict)
        result_dict["duration"] = output_dict["duration"]
        return result_dict

    def get_log_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict["scene_name"]
        image_id = data_dict["image_id"]
        cloud_id = data_dict["cloud_id"]
        message = f"{scene_name}, img: {image_id}, pcd: {cloud_id}"
        message += ", " + get_log_string(result_dict=result_dict)
        message += ", nCorr: {}".format(output_dict["corr_scores"].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict["scene_name"]
        image_id = data_dict["image_id"]
        cloud_id = data_dict["cloud_id"]

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f"{image_id}_{cloud_id}.npz")
        np.savez_compressed(
            file_name,
            image_file=data_dict["image_file"],
            depth_file=data_dict["depth_file"],
            # cloud_file=data_dict["cloud_file"],
            pcd_points=tensor_to_array(output_dict["pcd_points"]),
            pcd_points_f=tensor_to_array(output_dict["pcd_points_f"]),
            pcd_points_c=tensor_to_array(output_dict["pcd_points_c"]),
            img_num_nodes=output_dict["img_num_nodes"],
            pcd_num_nodes=output_dict["pcd_num_nodes"],
            img_node_corr_indices=tensor_to_array(output_dict["img_node_corr_indices"]),
            pcd_node_corr_indices=tensor_to_array(output_dict["pcd_node_corr_indices"]),
            img_node_corr_levels=tensor_to_array(output_dict["img_node_corr_levels"]),
            img_corr_points=tensor_to_array(output_dict["img_corr_points"]),
            pcd_corr_points=tensor_to_array(output_dict["pcd_corr_points"]),
            img_corr_pixels=tensor_to_array(output_dict["img_corr_pixels"]),
            pcd_corr_pixels=tensor_to_array(output_dict["pcd_corr_pixels"]),
            corr_scores=tensor_to_array(output_dict["corr_scores"]),
            gt_img_node_corr_indices=tensor_to_array(output_dict["gt_img_node_corr_indices"]),
            gt_pcd_node_corr_indices=tensor_to_array(output_dict["gt_pcd_node_corr_indices"]),
            gt_img_node_corr_overlaps=tensor_to_array(output_dict["gt_img_node_corr_overlaps"]),
            gt_pcd_node_corr_overlaps=tensor_to_array(output_dict["gt_pcd_node_corr_overlaps"]),
            gt_node_corr_min_overlaps=tensor_to_array(output_dict["gt_node_corr_min_overlaps"]),
            gt_node_corr_max_overlaps=tensor_to_array(output_dict["gt_node_corr_max_overlaps"]),
            transform=tensor_to_array(data_dict["transform"]),
            intrinsics=tensor_to_array(data_dict["intrinsics"]),
            # overlap=data_dict["overlap"],
        )

        # anpei visulization of 2d-3d registration
        import cv2 as cv
        import torch
        from vision3d.ops import apply_transform

        base_path = data_base_path
        save_path = data_save_path
        import os
        if os.path.exists(save_path) == False:
            os.mkdir(save_path)

        is_need_visulization = True
        # is_need_visulization = False
        if is_need_visulization:
            image_file = data_dict["image_file"]
            transform  = data_dict["transform"]
            intrinsics = data_dict["intrinsics"]
            pts        = data_dict["points"][0]
            pts_rgb    = data_dict["points_rgb"] #* 5

            '''
                1. visulize rgb/depth image
            '''
            # print("===> : ", base_path + image_file)
            rgb_image = cv.imread(base_path + image_file)
            '''
                note-0620: only used in kitti
            '''
            # h, w  = rgb_image.shape[0], rgb_image.shape[1]
            # image_re = np.zeros((480,640,3), dtype=np.float32)
            # image_re[0:h,0:640,0:3] = rgb_image[0:h,(608-320):(608+320),0:3]
            # rgb_image = image_re

            img_h = rgb_image.shape[0]
            img_w = rgb_image.shape[1]

            pts = apply_transform(pts, transform)
            pix = (torch.matmul(intrinsics, pts.T)).T
            pix = pix.cpu().numpy()
            dep = pix[:,2:3]
            pix = pix/dep
            pix = pix.astype(np.int)
            num = pix.shape[0]
            d_max, d_min = np.max(dep), np.min(dep)

            pts_image = np.zeros_like(rgb_image)
            for i in range(num):
                u = int(pix[i,0])
                v = int(pix[i,1])
                r = int(pts_rgb[i,0]*255)
                g = int(pts_rgb[i,1]*255)
                b = int(pts_rgb[i,2]*255)
                if ((u < 0) | (u >= img_w)):
                    continue
                if ((v < 0) | (v >= img_h)):
                    continue
                # draw circles --- r=2 is good :)
                cv.circle(pts_image, (int(u), int(v)), 2, (b,g,r), -1) # 3
                # draw dots --- not good
                # pts_image[int(v), int(u), 0] = b
                # pts_image[int(v), int(u), 1] = g
                # pts_image[int(v), int(u), 2] = r
            
            invalid_mask = (pts_image[:,:,0] == 0) \
                & (pts_image[:,:,1] == 0) \
                & (pts_image[:,:,2] == 0)
            pts_image[invalid_mask,:] = 255

            vis_rgb_pts_img = np.concatenate((rgb_image, pts_image), axis=1)
            vis_mix_img = cv.addWeighted(rgb_image, 0.5, pts_image, 0.5, 0)
            vis_mix_img = vis_rgb_pts_img * 1

            '''
                2. visulize 2d/3d corner points
            '''
            img_corr_pixels=tensor_to_array(output_dict["img_corr_pixels"])
            pcd_corr_pixels=tensor_to_array(output_dict["pcd_corr_pixels"])
            corr_scores=tensor_to_array(output_dict["corr_scores"])
            num_pts = img_corr_pixels.shape[0]
            for i in range(num_pts):
                # if corr_scores[i] >= 0:
                #     continue
                u = int(img_corr_pixels[i,0])
                v = int(img_corr_pixels[i,1])
                cv.circle(vis_rgb_pts_img, (int(v), int(u)), 1, (0,255,255), -1)
            
            pts_corr = apply_transform(
                torch.tensor(pcd_corr_pixels).cuda().float(), transform)
            pix_corr = (torch.matmul(intrinsics, pts_corr.T)).T
            pix_corr = pix_corr.cpu().numpy()
            dep_corr = pix_corr[:,2:3]
            pix_corr = pix_corr/dep_corr
            pix_corr = pix_corr.astype(np.int)
            for i in range(num_pts):
                # if corr_scores[i] >= 0:
                #     continue
                u = int(pix_corr[i,0])
                v = int(pix_corr[i,1])
                cv.circle(vis_rgb_pts_img, (int(u)+img_w, int(v)), 1, (0,255,255), -1)

            '''
                3. visulize 2d/3d point correspondence
            '''     
            for i in range(num_pts):
                # if corr_scores[i] >= 0:
                #     continue
                u_img = int(img_corr_pixels[i,1])
                v_img = int(img_corr_pixels[i,0])
                u_pts = int(pix_corr[i,0])
                v_pts = int(pix_corr[i,1])
                d = np.abs(u_img - u_pts) + np.abs(v_img - v_pts)
                th = 15
                if d > th:
                    cv.line(vis_rgb_pts_img, 
                        (u_img, v_img), (u_pts+img_w, v_pts), (0,0,255), 1)
                if d <= th:
                    cv.line(vis_rgb_pts_img, 
                        (u_img, v_img), (u_pts+img_w, v_pts), (0,255,0), 1)
            
            '''
                4. visulize PIR/IR in the image
            ''' 
            # res_string = "IR: "+ str(result_dict['IR'].cpu().numpy())
            res_string = "IR: "+ format(result_dict['IR'].cpu().numpy(), '.3f')
            cv.putText(vis_rgb_pts_img, res_string, 
                (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

            # print("img_corr_pixels: ", img_corr_pixels.shape)
            # print("pcd_corr_pixels: ", pcd_corr_pixels.shape)
            # print("result_dict")
            # print(result_dict)

            '''
                note-0731 --- 
                    additional visualization of i2p correspondences
            '''
            is_need_extra_visual = False
            if is_need_extra_visual:
                rgb_image_ext = rgb_image * 1
                for i in range(num_pts):
                    u_img = int(img_corr_pixels[i,1])
                    v_img = int(img_corr_pixels[i,0])
                    u_pts = int(pix_corr[i,0])
                    v_pts = int(pix_corr[i,1])
                    d = np.abs(u_img - u_pts) + np.abs(v_img - v_pts)
                    th = 15
                    if d <= th:
                        cv.circle(rgb_image_ext, (int(v_img), int(u_img)), 3, (0,255,255), -1)
                cv.imshow("rgb_image_ext", rgb_image_ext)

            # cv.imshow("vis_rgb_pts_img", vis_rgb_pts_img)
            # cv.imshow("vis_mix_img", vis_mix_img)
            # cv.waitKey(0)
            # assert 1==-1

            '''
                5. save the visulization image
            ''' 
            save_name = save_path + str(self.save_idx) + ".png"
            self.save_idx += 1
            cv.imwrite(save_name, vis_rgb_pts_img) # vis_mix_img vis_rgb_pts_img
            print("save image in the path: ", save_name)
            # assert 1==-1

def main():
    add_tester_args()
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == "__main__":
    main()