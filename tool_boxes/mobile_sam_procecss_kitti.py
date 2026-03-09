import os
import gradio as gr
import glob
import numpy as np
import torch
import cv2 as cv
import time as time
import matplotlib.pyplot as plt
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import ImageDraw
from MobileSAM.app.utils.tools import box_prompt, format_results, point_prompt
from MobileSAM.app.utils.tools_gradio import fast_process, fast_show_mask

base_path = "/media/anpei/DiskA/05_i2p_fewshot/tool_boxes/MobileSAM/"

def main_batch_process():
    '''
        a standard batch process to segment 2d image from the seven scenes dataset
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model
    sam_checkpoint = base_path + "weights/mobile_sam.pt"
    model_type = "vit_t"

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam = mobile_sam.to(device=device)
    mobile_sam.eval()

    # for image segmentation fast/online
    # mask_generator = SamAutomaticMaskGenerator(mobile_sam, points_per_side=8)
    # for point cloud segmentation slow/offline
    # mask_generator = SamAutomaticMaskGenerator(mobile_sam, points_per_side=16)
    mask_generator = SamAutomaticMaskGenerator(mobile_sam, points_per_side=32)

    '''
    note-0516
        anpei dataset path 
    '''
    base_data_dir  = "/media/anpei/DiskA/05_i2p_fewshot/data/"
    data_base_path = base_data_dir+"kitti/val_selection_cropped/image"
    path_num = 1

    for idx in range(path_num):
        spec_path = data_base_path + "/"
        img_paths = glob.glob(spec_path + "*.png")
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
                # if img_path[-9:-4] == "sampd":
                #     continue
                if img_path[-9:-4] == "depth":
                    continue
                print(img_path)

                time_st = time.time()
                image = cv.imread(img_path)
                w, h = image.shape[1], image.shape[0]
                nd_image = np.array(image)
                annotations = mask_generator.generate(nd_image)
                time_ed = time.time()
                print("==> cost time: ", time_ed-time_st)

                if isinstance(annotations[0], dict):
                    annotations = [annotation["segmentation"] for annotation in annotations]
                annotations = np.array(annotations)
                inner_mask = fast_show_mask(
                    annotations, plt.gca(), random_color=True,
                    bbox=None, retinamask=True, target_height=h, target_width=w)
                inner_mask = inner_mask[:,:,:3]

                # color normalization
                inner_mask = (inner_mask*255).astype(np.uint8)

                '''
                    save and visulize mobile segmentation results
                '''
                # target_path = img_path[:-4] + "_sampo.png" # for image fast segment
                target_path = img_path[:-4] + "_sampd.png" # point slow/offline segment
                cv.imwrite(target_path, inner_mask)
                
                # cv.imshow("inner_mask", inner_mask)
                # cv.imshow("image", image)
                # cv.waitKey(0)

if __name__ == '__main__':
    '''
        process a whole dataset
    '''
    main_batch_process()