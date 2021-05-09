import time
import datetime
import torch
import shutil

from saic_depth_completion.utils.meter import AggregatedMeter
from saic_depth_completion.utils.meter import Statistics as LossMeter
from saic_depth_completion.engine.val import validate
from utils.general_utlis import *

def train(
    args, model, trainloader, optimizer, val_loaders={}, scheduler=None, snapshoter=None, logger=None,
    epochs=100, init_epoch=0,  logging_period=10, metrics={}, tensorboard=None, tracker=None
):

    is_converge = False
    # move model to train mode
    model.train()
    logger.info(
        "Total number of params: {}".format(model.count_parameters())
    )
    loss_meter = LossMeter(maxlen=20)
    metrics_meter = AggregatedMeter(metrics, maxlen=20)
    logger.info(
        "Start training at {} epoch. Total number of epochs {}.".format(init_epoch, epochs)
    )

    num_batches = len(trainloader)
    mirror_score_list = []
    checkpoint_save_list = []
    start_time_stamp = time.time()
    for epoch in range(init_epoch, epochs):
        loss_meter.reset()
        metrics_meter.reset()
        # loop over dataset

        for it, batch in enumerate(trainloader):
            batch = model.preprocess(batch)
            pred = model(batch)
            loss = model.criterion(pred, batch["gt_depth"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), 1)

            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                post_pred = model.postprocess(pred)
                metrics_meter.update(post_pred, batch["gt_depth"])

            if (epoch * num_batches + it) % logging_period == 0:
                state = "ep: {}, it {}/{} -- loss {:.4f}({:.4f}) | ".format(
                    epoch, it, num_batches, loss_meter.median, loss_meter.global_avg
                )
                logger.info(state + metrics_meter.suffix)

            global_it = epoch * num_batches + it
            # validate and save checkpoint per snapshoter.period
            if snapshoter is not None and global_it % snapshoter.period == 0:
                snapshoter.save('snapshot_{}_{}'.format(epoch, global_it))

                mirror_rmse = validate(
                args, model, val_loaders, metrics, epoch=epoch, logger=logger,
                tensorboard=tensorboard, tracker=tracker, final_result=False,global_it=global_it
                )
                mirror_score_list.append(mirror_rmse)
                import os
                checkpoint_save_list.append(os.path.join(snapshoter.save_dir, "{}.pth".format('snapshot_{}_{}'.format(epoch, global_it))))

                if check_converge(score_list=mirror_score_list,check_freq=3):
                    final_checkpoint_src = checkpoint_save_list[-3]
                    final_checkpoint_dst = os.path.join(os.path.split(final_checkpoint_src)[0], "converge_{}".format(os.path.split(final_checkpoint_src)[-1]))
                    shutil.copy(final_checkpoint_src, final_checkpoint_dst)
                    is_converge = True
                    break
            if tensorboard is not None:
                tensorboard.update(
                    {k: v.global_avg for k, v in metrics_meter.meters.items()}, tag="train", iter=global_it
                )
            
            
        if is_converge:
            break
            


        state = "ep: {}, it {}/{} -- loss {:.4f}({:.4f}) | ".format(
            epoch, it, num_batches, loss_meter.median, loss_meter.global_avg
        )

        logger.info(state + metrics_meter.suffix)



    if snapshoter is not None:
        snapshoter.save('last_snapshot_{}_{}'.format(epoch, global_it))
        mirror_rmse = validate(
                    args, model, val_loaders, metrics, epoch=epoch, logger=logger,
                    tensorboard=tensorboard, tracker=tracker, final_result=True,global_it=global_it
        )
        mirror_score_list.append(mirror_rmse)
        checkpoint_save_list.append(os.path.join(snapshoter.save_dir, "{}.pth".format('last_snapshot_{}_{}'.format(epoch, global_it))))


    total_time = str(datetime.timedelta(seconds=time.time() - start_time_stamp))

    logger.info(
        "Training finished! Total spent time: {}.".format(total_time)
    )