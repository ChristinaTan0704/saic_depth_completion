import time
import datetime
import torch
from tqdm import tqdm

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from utils.mirror3d_metrics import Mirror3dEval
import cv2

def validate(
        args, model, val_loaders, metrics, epoch=0, logger=None, tensorboard=None, tracker=None, final_result=False,global_it=0
):
    mirror3d_eval = Mirror3dEval(args.refined_depth, logger, Input_tag="RGBD", method_tag="saic",dataset_root=args.coco_val_root)
    model.eval()
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    for subset, loader in val_loaders.items():
        logger.info(
            "Validate: ep: {}, subset -- {}. Total number of batches: {}.".format(epoch, subset, len(loader))
        )

        metrics_meter.reset()
        # loop over dataset
        for batch in tqdm(loader):
            batch = model.preprocess(batch)
            pred = model(batch)
            
            with torch.no_grad():
                post_pred = model.postprocess(pred)
                metrics_meter.update(post_pred, batch["gt_depth"])
                pred_depth = post_pred.squeeze().cpu()
                gt_depth_path = batch["gt_depth_path"][0]
                gt_depth = cv2.resize(cv2.imread(gt_depth_path, cv2.IMREAD_ANYDEPTH), (pred_depth.shape[1], pred_depth.shape[0]), 0, 0, cv2.INTER_NEAREST) / args.depth_shift
                mirror3d_eval.compute_and_update_mirror3D_metrics(pred_depth, args.depth_shift, batch["color_img_path"][0], batch["rawD_path"][0], batch["gt_depth_path"][0], batch["mask_path"][0])
                if final_result:
                    mirror3d_eval.save_result(args.log_directory, pred_depth, args.depth_shift, batch["color_img_path"][0], batch["rawD_path"][0], batch["gt_depth_path"][0], batch["mask_path"][0])

        mirror3d_eval.print_mirror3D_score()
        state = "Validate: global_it: {}, subset -- {} | ".format(global_it, subset)
        logger.info(state + metrics_meter.suffix)
        metric_state = {k: v.global_avg for k, v in metrics_meter.meters.items()}
        metric_state["mirror_rmse"] = torch.tensor((mirror3d_eval.m_nm_all_refD/ mirror3d_eval.ref_cnt)[0])
        if tensorboard is not None:
            tensorboard.update(metric_state, tag=subset, iter=global_it)

        if tracker is not None:
            tracker.update(subset, metric_state)
        
        mirror_rmse = (mirror3d_eval.m_nm_all_refD/ mirror3d_eval.ref_cnt)[0]
        return mirror_rmse