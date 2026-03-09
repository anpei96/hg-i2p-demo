import time

'''
    please revise make_cfg for training in other scenes
'''
# from config.seven_scenes_config import make_cfg
# from config.rgbd_scenes_config import make_cfg
# from config.scannet_scenes_config import make_cfg
# from config.self_v1_scenes_config import make_cfg

# from config.seven_scenes_config_ours import make_cfg
# from config.seven_scenes_config_kpts import make_cfg
from config.kitti_scenes_config import make_cfg
'''
'''
'''
    please choose different i2p registration method
'''
# from model_keypoints.model_semi_bpnp import create_solver_model
from model_keypoints.model_bpnp_learning import create_solver_model
from dataset.dataset_kitti import train_valid_data_loader
# from dataset.dataset import train_valid_data_loader
from loss.loss_bpnp_circle import KeypointsLoss, EvalFunction
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
            load i2p registration model
        '''
        model = create_solver_model(cfg)
        model = self.register_model(model)

        '''
            load pre-trained i2p registration model
        '''
        base_path = "/media/anpei/DiskA/05_i2p_fewshot/model_zoos_kpts/"
        ckpt_path = base_path + "base-20-kitti.pth"
        self.load(ckpt_path)
        self._max_epoch = 30

        # optimizer, scheduler
        optimizer = build_optimizer(model, cfg)
        self.register_optimizer(optimizer)
        scheduler = build_scheduler(optimizer, cfg)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = KeypointsLoss(cfg)
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
