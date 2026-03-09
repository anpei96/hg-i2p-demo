import time

'''
    please revise make_cfg for training in other scenes
'''
# from config.seven_scenes_config import make_cfg
# from config.rgbd_scenes_config import make_cfg
# from config.scannet_scenes_config import make_cfg
# from config.self_v1_scenes_config import make_cfg
# from config.kitti_scenes_config import make_cfg
from config.seven_scenes_config_ours import make_cfg
# from config.seven_scenes_config_kpts import make_cfg
'''
'''
'''
    please choose different i2p registration method
'''
# from model_matr.model import create_model
# from dataset.dataset import train_valid_data_loader
# from loss.loss import EvalFunction, OverallLoss

# from model_fewshot.model import create_model     # top-i2p
from model_fewshot.model_plus import create_model  # top-i2p++
from dataset.dataset import train_valid_data_loader
# from loss.loss_exp import EvalFunction, OverallLoss # top-i2p
from loss.loss_exp_plus import EvalFunction, OverallLoss # top-i2p++

# from model_keypoints.model_kpts_learning import create_model
# from dataset.dataset import train_valid_data_loader
# from loss.loss_kpts import EvalFunction, OverallLoss

# from model_matr_aug.model import create_model
# from dataset.dataset import train_valid_data_loader
# from loss.loss_aug import EvalFunction, OverallLoss
# from config.seven_scenes_config_aug import make_cfg
'''
'''
from vision3d.engine import EpochBasedTrainer
from vision3d.utils.optimizer import build_optimizer, build_scheduler

class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg)
        loading_time = time.time() - start_time
        self.log(f"Data loader created: {loading_time:.3f}s collapsed.", level="DEBUG")
        self.log(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(train_loader, val_loader)
    
        '''
            load different i2p registration model
        '''
        model = create_model(cfg)
        model = self.register_model(model)

        '''
            load pre-trained i2p registration model
        '''
        base_path = "/media/anpei/DiskA/05_i2p_fewshot/model_zoos/"
        # ckpt_path = base_path + "epoch-5-chess.pth"
        # ckpt_path = base_path + "epoch-7-kitchen.pth"
        # ckpt_path = base_path + "baseline-kitchen.pth"
        # ckpt_path = base_path + "baseline-office.pth"
        # self.load(ckpt_path)
        # self._max_epoch = 25
        base_path = "/media/anpei/DiskA/05_i2p_fewshot/model_zoos_plus/"
        # ckpt_path = base_path + "base-office-20.pth"
        # ckpt_path = base_path + "base-chess-20.pth"
        # ckpt_path = base_path + "base-kitchen-30.pth"
        # ckpt_path = base_path + "our-50-tr-align-act-tf-kitchen.pth"
        ckpt_path = base_path + "our-30-tr-align-act-tf-office.pth"
        self.load(ckpt_path)
        # self._max_epoch = 50
        # self._max_epoch = 30
        self._max_epoch = 60

        # optimizer, scheduler
        optimizer = build_optimizer(model, cfg)
        self.register_optimizer(optimizer)
        scheduler = build_scheduler(optimizer, cfg)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg)
        self.eval_func = EvalFunction(cfg)

    def train_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        # result_dict = self.eval_func(data_dict, output_dict)
        # loss_dict.update(result_dict)
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(data_dict, output_dict)
        result_dict = self.eval_func(data_dict, output_dict)
        loss_dict.update(result_dict)
        return output_dict, loss_dict

def main():
    cfg = make_cfg()
    trainer = Trainer(cfg)
    trainer.run_only_train()

if __name__ == "__main__":
    main()
