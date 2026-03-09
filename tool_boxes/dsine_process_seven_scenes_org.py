import os
import sys
import glob
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import sys
import time
import DSINE.utils.utils as utils
import DSINE.projects.dsine.config as config
from DSINE.utils.projection import intrins_from_fov, intrins_from_txt

if __name__ == '__main__':
    device = torch.device('cuda')
    args = config.get_args(test=True)
    args.ckpt_path = "/media/anpei/DiskA/05_i2p_fewshot/tool_boxes/DSINE/dsine.pt"
    assert os.path.exists(args.ckpt_path)

    if args.NNET_architecture == 'v00':
        from DSINE.models.dsine.v00 import DSINE_v00 as DSINE
    elif args.NNET_architecture == 'v01':
        from DSINE.models.dsine.v01 import DSINE_v01 as DSINE
    elif args.NNET_architecture == 'v02':
        from DSINE.models.dsine.v02 import DSINE_v02 as DSINE
    elif args.NNET_architecture == 'v02_kappa':
        from DSINE.models.dsine.v02_kappa import DSINE_v02_kappa as DSINE
    else:
        raise Exception('invalid arch')

    model = DSINE(args).to(device)
    model = utils.load_checkpoint(args.ckpt_path, model)
    model.eval()

    '''
    note-0516
        anpei dataset path 
    '''
    data_dir = "/media/anpei/DiskA/05_i2p_fewshot/data/"
    base_path = data_dir+"kitti/val_selection_cropped/image"

    fx = 721.5377 
    fy = 721.5377 
    cx = 596.5593
    cy = 149.8540
    intrins = torch.tensor([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=torch.float32, device=device).unsqueeze(0)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    for idx in range(1):
        idx = idx + 1
        spec_path = base_path + "/"
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
                if img_path[-9:-4] == "sampd":
                    continue
                if img_path[-9:-4] == "depth":
                    continue
                print(img_path)

                time_st = time.time()
                ext = os.path.splitext(img_path)[1]
                img = Image.open(img_path).convert('RGB')
                img = np.array(img).astype(np.float32) / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

                # pad input
                _, _, orig_H, orig_W = img.shape
                lrtb = utils.get_padding(orig_H, orig_W)
                img = F.pad(img, lrtb, mode="constant", value=0.0)
                img = normalize(img)

                intrins[:, 0, 2] += lrtb[0]
                intrins[:, 1, 2] += lrtb[2]

                pred_norm = model(img, intrins=intrins)[-1]
                pred_norm = pred_norm[:, :, lrtb[2]:lrtb[2]+orig_H, lrtb[0]:lrtb[0]+orig_W]
                pred_norm_raw = (pred_norm).detach().cpu().permute(0, 2, 3, 1).numpy()
                time_ed = time.time()
                print("==> normal estimation time: ", time_ed-time_st)

                # save to output folder
                # NOTE: by saving the prediction as uint8 png format, you lose a lot of precision
                # if you want to use the predicted normals for downstream tasks, we recommend saving them as float32 NPY files
                target_path = img_path[:-4] + "_dsine.png"
                pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
                pred_norm = (((pred_norm + 1) * 0.5) * 255).astype(np.uint8)
                im = Image.fromarray(pred_norm[0,...])
                im.save(target_path)

                target_path = img_path[:-4] + "_dsine.npy"
                #np.save(target_path, pred_norm_raw)

                
                
                
