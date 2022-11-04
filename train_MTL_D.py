import os
import datetime
from pickle import FALSE, TRUE

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.hrnet_MTL_D import HRnet
from nets.hrnet_training import (get_lr_scheduler, set_optimizer_lr,
                                 weights_init)
from utils.callbacks_multi import LossHistory, EvalCallback
from utils.dataloader_multi import SegmentationDataset, seg_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit_multi import fit_one_epoch
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
from nets.hrnet_training import (CE_Loss, Dice_loss, Focal_Loss,
                                     weights_init)
from tqdm import tqdm
from utils.utils import get_lr
from utils.utils_metrics_multi import f_score
import itertools

if __name__ == "__main__":
    Cuda            = True
    distributed     = False
    sync_bn         = False
    fp16            = True
    num_classes     = [7,2]
    backbone        = "hrnetv2_w32"
    pretrained      = False
    model_path      = "model_data/hrnetv2_w32_weights_voc.pth"
    input_shape     = [480, 480]
    Init_Epoch          = 0
    Freeze_Epoch        = 15
    Freeze_batch_size   = 8
    UnFreeze_Epoch      = 200
    Unfreeze_batch_size = 8
    Freeze_Train        = False
    Init_lr             = 5e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = 'exp'
    save_period         = 10
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 5
    VOCdevkit_path  = 'VOCdevkit'
    dice_loss       = False
    focal_loss      = False
    cls_weights = np.array([[1, 1, 1, 1, 1, 1, 1], [1, 1]], object)
    num_workers     = 4

    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)  
            dist.barrier()
        else:
            download_weights(backbone)

    model   = HRnet(num_classes=num_classes, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None
        
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        awl = AutomaticWeightedLoss(2).cuda()

        optimizer = optim.Adam([
                {'params': model.parameters(), 'lr': Init_lr_fit, 'betas': (momentum, 0.999), 'weight_decay': weight_decay},
                {'params': awl.parameters(), 'lr': 1e-2, 'weight_decay': 0}
            ])

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please expand the dataset.")
        
        train_dataset   = SegmentationDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = SegmentationDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
    
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler)

        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                for param in model.backbone.parameters():
                    param.requires_grad = True
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=train_sampler)
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = seg_dataset_collate, sampler=val_sampler)

                UnFreeze_flag   = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            if True:
                    total_loss      = 0
                    total_loss_e      = 0
                    total_loss_d      = 0
                    total_f_score   = 0

                    val_loss        = 0
                    val_loss_e        = 0
                    val_loss_d        = 0
                    val_f_score     = 0

                    if local_rank == 0:
                        print('Start Train')
                        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}',postfix=dict,mininterval=0.3)
                    model_train.train()
                    for iteration, batch in enumerate(gen):
                        if iteration >= epoch_step: 
                            break
                        imgs, pngs_e, pngs_d, labels_e, labels_d = batch
                        with torch.no_grad():
                            cls_weights_1 = np.array(cls_weights[0], np.float32)
                            cls_weights_2 = np.array(cls_weights[1], np.float32)
                            weights_1 = torch.from_numpy(cls_weights_1)
                            weights_2 = torch.from_numpy(cls_weights_2)

                            if Cuda:
                                imgs    = imgs.cuda(local_rank)
                                pngs_e  = pngs_e.cuda(local_rank) 
                                pngs_d  = pngs_d.cuda(local_rank)              
                                labels_e = labels_e.cuda(local_rank)
                                labels_d = labels_d.cuda(local_rank)
                                weights_1 = weights_1.cuda(local_rank)
                                weights_2 = weights_2.cuda(local_rank)

                        optimizer.zero_grad()
                        if not fp16:
                            outputs = model_train(imgs)
                            if focal_loss:
                                loss_e = Focal_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                                loss_d = Focal_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])
                            else:
                                loss_e = CE_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                                loss_d = CE_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])

                            if dice_loss:
                                main_dice_e = Dice_loss(outputs[0], labels_e)
                                main_dice_d = Dice_loss(outputs[1], labels_d)
                                loss_e      = loss_e + main_dice_e
                                loss_d      = loss_d + main_dice_d
                                
                            with torch.no_grad():
                                _f_score_e = f_score(outputs[0], labels_e)
                                _f_score_d = f_score(outputs[1], labels_d)

                            loss = awl(loss_e, loss_d)
                            loss.backward()
                            optimizer.step()

                        else:
                            from torch.cuda.amp import autocast
                            with autocast():
                                outputs = model_train(imgs)
                                if focal_loss:
                                    loss_e = Focal_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                                    loss_d = Focal_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])
                                else:
                                    loss_e = CE_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                                    loss_d = CE_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])                                

                                if dice_loss:
                                    main_dice_e = Dice_loss(outputs[0], labels_e)
                                    main_dice_d = Dice_loss(outputs[1], labels_d)
                                    loss_e      = loss_e + main_dice_e
                                    loss_d      = loss_d + main_dice_d

                                with torch.no_grad():
                                    _f_score_e = f_score(outputs[0], labels_e)
                                    _f_score_d = f_score(outputs[1], labels_d)

                            loss = awl(loss_e, loss_d)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            
                        total_loss      += loss_e.item()
                        total_loss_e    += loss_e.item()
                        total_loss      += loss_d.item()
                        total_loss_d    += loss_d.item()
                        total_f_score   += _f_score_e.item()
                        total_f_score   += _f_score_d.item()
                        
                        if local_rank == 0:
                            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                                'f_score'   : total_f_score / (iteration + 1),
                                                'lr'        : get_lr(optimizer)})
                            pbar.update(1)

                    if local_rank == 0:
                        pbar.close()
                        print('Finish Train')
                        print('Start Validation')
                        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{UnFreeze_Epoch}',postfix=dict,mininterval=0.3)

                    model_train.eval()
                    for iteration, batch in enumerate(gen_val):
                        if iteration >= epoch_step_val:
                            break
                        imgs, pngs_e, pngs_d, labels_e, labels_d = batch
                        with torch.no_grad():
                            cls_weights_1 = np.array(cls_weights[0], np.float32)
                            cls_weights_2 = np.array(cls_weights[1], np.float32)
                            weights_1 = torch.from_numpy(cls_weights_1)
                            weights_2 = torch.from_numpy(cls_weights_2)

                            if Cuda:
                                imgs    = imgs.cuda(local_rank)
                                pngs_e  = pngs_e.cuda(local_rank) 
                                pngs_d  = pngs_d.cuda(local_rank)              
                                labels_e = labels_e.cuda(local_rank)
                                labels_d = labels_d.cuda(local_rank)
                                weights_1 = weights_1.cuda(local_rank)
                                weights_2 = weights_2.cuda(local_rank) 

                            outputs     = model_train(imgs)
                            if focal_loss:
                                loss_e = Focal_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                                loss_d = Focal_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])
                            else:
                                loss_e = CE_Loss(outputs[0], pngs_e, weights_1, num_classes = num_classes[0])
                                loss_d = CE_Loss(outputs[1], pngs_d, weights_2, num_classes = num_classes[1])

                            if dice_loss:
                                main_dice_e = Dice_loss(outputs[0], labels_e)
                                main_dice_d = Dice_loss(outputs[1], labels_d)
                                loss_e      = loss_e + main_dice_e
                                loss_d      = loss_d + main_dice_d
                            _f_score_e = f_score(outputs[0], labels_e)
                            _f_score_d = f_score(outputs[1], labels_d)

                            val_loss_e  += loss_e.item()
                            val_loss_d  += loss_d.item()
                            val_f_score += _f_score_e.item()
                            val_f_score += _f_score_d.item()
                            
                        if local_rank == 0:
                            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                                'f_score'   : val_f_score / (iteration + 1),
                                                'lr'        : get_lr(optimizer)})
                            pbar.update(1)
                            
                    if local_rank == 0:
                        pbar.close()
                        print('Finish Validation')
                        loss_history.append_loss(epoch + 1, total_loss / epoch_step, total_loss_e / epoch_step, total_loss_d / epoch_step, val_loss / epoch_step_val, val_loss_e / epoch_step_val, val_loss_d / epoch_step_val)
                        eval_callback.on_epoch_end(epoch + 1, model_train)
                        print('Epoch:'+ str(epoch + 1) + '/' + str(UnFreeze_Epoch))
                        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
                        print('Element Loss: %.3f || Element Val Loss: %.3f || Defect Loss: %.3f || Defect Val Loss: %.3f ' % (total_loss_e / epoch_step, val_loss_e / epoch_step_val, total_loss_d / epoch_step, val_loss_d / epoch_step_val))
                        
                        if (epoch + 1) % save_period == 0 or epoch + 1 == UnFreeze_Epoch:
                            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

                        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
                            print('Save best model to best_epoch_weights.pth')
                            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
                            
                        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))

            if distributed:
                dist.barrier()

            for p in awl.parameters():
                print('aw', p)

        if local_rank == 0:
            loss_history.writer.close()
