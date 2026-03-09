from typing import Union

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from vision3d.layers import ConvBlock, build_act_layer
from .image_backbone  import BasicBlock

class CovDiFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.zip_num = 128
        self.zip_fea_2d = nn.Sequential(
            ConvBlock(128, self.zip_num, 3, 1, 1, conv_cfg="Conv2d", norm_cfg="BatchNorm", act_cfg="ReLU"))
        self.zip_fea_3d = nn.Sequential(nn.Linear(128, self.zip_num), nn.ReLU())

    def forward(self, data_dict, img_feats_f, pcd_feats_f):
        '''
            please note that the batch size is only one

            img_feats_f:  torch.Size([1, 128, 480, 640])
            pcd_feats_f:  torch.Size([22432, 128])
        '''
        
        '''
            step one: construct covariance matrix
        '''
        num_channel = self.zip_num                             # C
        img_feats_f_zip = self.zip_fea_2d(img_feats_f) 
        pcd_feats_f_zip = self.zip_fea_3d(pcd_feats_f) 

        tmp_img_f = img_feats_f_zip.reshape((num_channel,-1)) # [C,H*W]
        tmp_pcd_f = pcd_feats_f_zip.t()                       # [C,N]
        cov_img_mat = torch.cov(tmp_img_f) # [C,C]
        cov_pcd_mat = torch.cov(tmp_pcd_f) # [C,C]

        '''
            step two: covariance matrix interaction via covariance transformer
                      https://zhuanlan.zhihu.com/p/662777298
        '''
        # sc_ip = torch.matmul(cov_img_mat, cov_pcd_mat.t()) / self.zip_num
        # sc_pi = torch.matmul(cov_pcd_mat, cov_img_mat.t()) / self.zip_num
        eye = torch.eye(num_channel).cuda() * 1e-6
        sc_ip = torch.matmul(cov_img_mat, (cov_pcd_mat+eye).inverse()) / self.zip_num
        sc_pi = torch.matmul(cov_pcd_mat, (cov_img_mat+eye).inverse()) / self.zip_num

        at_ip = torch.nn.functional.softmax(sc_pi, dim=-1)
        at_pi = torch.nn.functional.softmax(sc_ip, dim=-1)

        # tmp_img_f = torch.matmul(at_ii, tmp_img_f)*0.5 + \
        #     torch.matmul(at_ip, tmp_img_f)*0.5
        # tmp_pcd_f = torch.matmul(at_pp, tmp_pcd_f)*0.5 + \
        #     torch.matmul(at_pi, tmp_pcd_f)*0.5
        tmp_img_f = torch.matmul(at_ip, tmp_img_f)
        tmp_pcd_f = torch.matmul(at_pi, tmp_pcd_f)
        
        h, w = img_feats_f.size(2), img_feats_f.size(3)
        tmp_img_f = tmp_img_f.reshape((-1, h, w)).unsqueeze(0)
        tmp_pcd_f = tmp_pcd_f.t()
        
        '''
            step three: feature fusion and merge
        '''
        tau = 0.1
        img_feats_f = tmp_img_f*tau + img_feats_f*(1-tau)
        pcd_feats_f = tmp_pcd_f*tau + pcd_feats_f*(1-tau)

        return img_feats_f, pcd_feats_f

        
