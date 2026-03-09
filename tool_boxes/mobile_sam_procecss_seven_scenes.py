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
data_path = "/media/anpei/DiskA/05_i2p_fewshot/data/7Scenes/data/"

def main_single_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained model
    sam_checkpoint = base_path + "weights/mobile_sam.pt"
    model_type = "vit_t"

    mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    mobile_sam = mobile_sam.to(device=device)
    mobile_sam.eval()

    mask_generator = SamAutomaticMaskGenerator(mobile_sam, points_per_side=8)
    # predictor = SamPredictor(mobile_sam)

    '''
        segment_everything pipeline
    '''
    png_file_path = data_path + "chess/seq-01/color_025.png"
    image = cv.imread(png_file_path)
    w = image.shape[1]
    h = image.shape[0]
    # image = cv.resize(image, dsize=(w//2,h//2))
    # w = image.shape[1]
    # h = image.shape[0]

    with torch.no_grad():
        nd_image = np.array(image)
        print("nd_image: ", nd_image.shape)
        time_st = time.time()
        annotations = mask_generator.generate(nd_image)
        time_ed = time.time()
        print("==> cost time: ", time_ed-time_st)

        time_st = time.time()
        annotations = mask_generator.generate(nd_image)
        time_ed = time.time()
        print("==> cost time: ", time_ed-time_st)

        time_st = time.time()
        annotations = mask_generator.generate(nd_image)
        time_ed = time.time()
        print("==> cost time: ", time_ed-time_st)

        time_st = time.time()
        annotations = mask_generator.generate(nd_image)
        time_ed = time.time()
        print("==> cost time: ", time_ed-time_st)
    
    if isinstance(annotations[0], dict):
        annotations = [annotation["segmentation"] for annotation in annotations]
    
    
    '''
        visulization pipeline v2 (only visulize top-4 class)
    '''
    is_use_v2 = False
    if is_use_v2:
        annotations = np.array(annotations)
        num_all_class = annotations.shape[0]
        h, w = annotations.shape[1], annotations.shape[2]
        areas = np.sum(annotations, axis=(1, 2))
        sorted_indices = np.argsort(areas)[::1]
        annotations = annotations[sorted_indices]
        res_img = np.zeros((h,w,3), dtype=np.uint8)

        num_of_top_class = 8
        seg_map_list = []
        for i in range(num_of_top_class):
            if i < num_all_class:
                x = np.expand_dims(annotations[-1-i,:,:], axis=-1)
                r = 25*i 
                g = 250-25*i 
                b = 25*i 
            else:
                continue
            
            is_save = True
            if i > 1:
                '''
                    remove redunent segmentation regions
                '''
                mask = annotations[-1-i+1,:,:] * annotations[-1-i,:,:]
                if np.sum(mask) > 1000:
                    is_save = False
            
            if is_save == True:
                y = np.concatenate((x*r,x*g,x*b), axis=2) # [h,w,3]
                y = y.astype(np.uint8)
                seg_map_list.append(y) # store binary map
                res_img += y

        cv.imshow("res_img", res_img)
        cv.imshow("image", image)
        cv.waitKey(0)

    '''
        visulization pipeline v1
    '''
    is_use_v1 = True
    if is_use_v1:
        annotations = np.array(annotations)
        print("==> segmentation classes: ", annotations.shape[0])
        print("==> segmentation height: ", annotations.shape[1])
        print("==> segmentation width: ", annotations.shape[2])
        inner_mask = fast_show_mask(
            annotations,
            plt.gca(),
            random_color=True,
            bbox=None,
            retinamask=True,
            target_height=h,
            target_width=w)
        
        cv.imshow("inner_mask", inner_mask)
        cv.imshow("image", image)
        cv.waitKey(0)

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
    mask_generator = SamAutomaticMaskGenerator(mobile_sam, points_per_side=16)

    '''
    note-0516
        anpei dataset path 
    '''
    data_dir = "/media/anpei/DiskA/05_i2p_fewshot/data/"
    data_base_path = data_dir+"7Scenes/data/chess"
    path_num = 6
    data_base_path = data_dir+"7Scenes/data/fire"
    path_num = 4
    data_base_path = data_dir+"7Scenes/data/heads"
    path_num = 2
    data_base_path = data_dir+"7Scenes/data/office"
    path_num = 10
    data_base_path = data_dir+"7Scenes/data/pumpkin"
    path_num = 8
    data_base_path = data_dir+"7Scenes/data/redkitchen"
    path_num = 14
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
                # cv.imwrite(target_path, inner_mask)
                
                # cv.imshow("inner_mask", inner_mask)
                # cv.imshow("image", image)
                # cv.waitKey(0)

if __name__ == '__main__':
    '''
        test single image
    '''
    # main_single_test()
    '''
        process a whole dataset
    '''
    main_batch_process()