import torch

import argparse

from saic_depth_completion.data.datasets.matterport import Matterport
from saic_depth_completion.data.datasets.nyuv2_test import NyuV2Test
from saic_depth_completion.engine.inference import inference
from saic_depth_completion.utils.tensorboard import Tensorboard
from saic_depth_completion.utils.logger import setup_logger
from saic_depth_completion.utils.experiment import setup_experiment
from saic_depth_completion.utils.snapshoter import Snapshoter
from saic_depth_completion.modeling.meta import MetaModel
from saic_depth_completion.config import get_default_config
from saic_depth_completion.data.collate import default_collate
from saic_depth_completion.metrics import Miss, SSIM, DepthL2Loss, DepthL1Loss, DepthRel
import time
import os

def main():
    parser = argparse.ArgumentParser(description="Some training params.")

    parser.add_argument('--model_name',                type=str,   help='model name', default='saic')

    # Input source config
    parser.add_argument('--coco_val',                  type=str,   help='coco json path', default='')
    parser.add_argument('--coco_train',                type=str,   help='coco json path', default='')
    parser.add_argument('--coco_train_root',           type=str,   help='coco data root', default="")
    parser.add_argument('--coco_val_root',             type=str,   help='coco data root', default="")

    # Input format config
    parser.add_argument('--refined_depth',             action='store_true',  help='using coco input format or not')
    parser.add_argument('--mesh_depth',                action='store_true',  help='using coco input format or not')
    parser.add_argument('--coco_focal_len',            type=int,   help='nyu : 519', default=519)
    parser.add_argument('--depth_shift',               type=int,   help='nyu, scannet : 1000, m3d : 4000', default=1000) 
    parser.add_argument('--input_height',              type=int,   help='input height', default=480)
    parser.add_argument('--input_width',               type=int,   help='input width',  default=640)
    

    # test and save
    parser.add_argument('--resume_checkpoint_path',    type=str,   help='path to a checkpoint to load', default="")
    parser.add_argument('--log_directory',             type=str,   help='training output folder', default='output')
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', dest='epoch', default=100)
    parser.add_argument('--output_save_folder',        type=str,   help='output_main_folder only use during inference', default='exp_result')


    parser.add_argument(
        "--debug", dest="debug", type=bool, default=False, help="Setup debug mode"
    )
    parser.add_argument(
        "--postfix", dest="postfix", type=str, default="", help="Postfix for experiment's name"
    )
    parser.add_argument(
        "--default_cfg", dest="default_cfg", type=str, default="LRN", help="Default config"
    )
    parser.add_argument(
        "--config_file", default="./configs/lrn/LRN_efficientnet-b4_lena.yaml", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument( # TODO
        "--test_batch_size", default=1, type=int, 
    )

    args = parser.parse_args()

    print("################# check point ##################")

    cfg = get_default_config(args.default_cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MetaModel(cfg, device)


    time_tag = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    output_folder = os.path.join(args.output_save_folder , "SAIC_infer_{}".format(time_tag))
    os.makedirs(output_folder, exist_ok=True)
    args.log_directory = output_folder

    logger = setup_logger(args.log_directory)
    logger.info("checkpoint {}".format(args.resume_checkpoint_path))

    snapshoter = Snapshoter(model, logger=logger)
    snapshoter.load(args.resume_checkpoint_path)

    metrics = {
        'mse': DepthL2Loss(),
        'mae': DepthL1Loss(),
        'd105': Miss(1.05),
        'd110': Miss(1.10),
        'd125_1': Miss(1.25),
        'd125_2': Miss(1.25**2),
        'd125_3': Miss(1.25**3),
        'rel': DepthRel(),
        'ssim': SSIM(),
    }

    test_datasets = {
        "test_matterport": Matterport(root=args.coco_val_root, coco_path=args.coco_val, refined_depth=args.refined_depth,split="test",args=args),

    }
    test_loaders = {
        k: torch.utils.data.DataLoader(
            dataset=v,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            collate_fn=default_collate
        )
        for k, v in test_datasets.items()
    }

    inference(
        args,
        model,
        test_loaders,
        save_dir=args.log_directory,
        logger=logger,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()