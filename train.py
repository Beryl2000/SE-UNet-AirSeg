import numpy as np
import os
import torch
import re
import time
import cc3d
import skimage.measure as measure
import SimpleITK as sitk
import bisect
import shutil
from torch.utils.data import DataLoader
from SE_UNet import DMR_UNet
from data import AirwayHMData, OnlineHMData,OnlineHMData3,AirwayHMData3, CropSegData, SegValCropData
from save_gradients import save_gradients_tw
from weight_br import save_s2_pred,save_weight_break
from metrics import *
from util import *
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import datetime



def double_threshold_iteration(pred, h_thresh, l_thresh):
    neigb = np.array(
    [[-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, -1, 0], [0, 1, 0], [1, -1, 0], [1, 0, 0], [1, 1, 0], [-1, -1, -1],
    [-1, 0, -1], [-1, 1, -1], [0, -1, -1], [0, 0, -1], [0, 1, -1], [1, -1, -1], [1, 0, -1], [1, 1, -1],
    [-1, -1, 1], [-1, 0, 1], [-1, 1, 1], [0, -1, 1], [0, 0, 1], [0, 1, 1], [1, -1, 1], [1, 0, 1],[1, 1, 1]])
    h, w,z = pred.shape
    pred = np.array(pred*255, dtype=np.float32)
    bin = np.where(pred >= h_thresh*255, 255, 0).astype(np.float32)
    gbin = bin.copy()
    gbin_pre = gbin-1
    while(gbin_pre.all() != gbin.all()):
        gbin_pre = gbin
        for i in range(h):
            for j in range(w):
                for k in range(z):
                    if gbin[i][j][k] == 0 and pred[i][j][k] < h_thresh*255 and pred[i][j][k] >= l_thresh*255:
                        for n in range(0, 26):  # 26领域
                            inn = i + neigb[n, 0]
                            jnn = j + neigb[n, 1]
                            knn = k + neigb[n, 2]
                            if gbin[max(min(inn,h-1),0)][max(min(jnn,w-1),0)][max(min(knn,z-1),0)]:
                                gbin[i][j][k] = 255
                                break

    return gbin/255

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = ((iflat) * tflat).sum()

    return 1 - ((2. * intersection + smooth) / ((iflat).sum() + (tflat).sum() + smooth))

def general_union_loss_lib(pred, target, weight):
    smooth = 1.0
    alpha = 0.2  # alpha=0.1 in stage1 and 0.2 in stage2
    beta = 1 - alpha
    sigma1 = 0.0001
    sigma2 = 0.0001
    weight_i = target * sigma1 + (1 - target) * sigma2
    intersection = (weight * ((pred + weight_i) ** 0.7) * target).sum()
    intersection2 = (weight * (alpha * pred + beta * target)).sum()
    return 1 - (intersection + smooth) / (intersection2 + smooth)

def br_loss(pred, target, skel,weight):
    smooth = 1.0
    target=skel
    pred=pred*skel
    intersection = (weight * pred* target).sum()
    intersection2 = (weight * (pred + target)).sum()
    return 1 - (intersection + smooth) / (intersection2 + smooth)

def D_Wi(pred_en, pred_de):
    # #计算动态权重
    gamma = 2
    alpha = 0.9

    pred_en_numpy = pred_en.cpu().detach().numpy()
    pred_de_numpy = pred_de.cpu().detach().numpy()

    smooth = 1e-5

    loss11 = (1 - pred_en_numpy)**gamma * np.log10(pred_en_numpy + smooth)
    loss10 = (pred_en_numpy)**gamma * np.log10(1 - pred_en_numpy + smooth)
    dynamic_weight1 = -alpha * loss11 - (1 - alpha) * loss10
    dynamic_weight1[dynamic_weight1 > 1] = 1

    loss21 = (1 - pred_de_numpy)**gamma * np.log10(pred_de_numpy + smooth)
    loss20 = (pred_de_numpy)**gamma * np.log10(1 - pred_de_numpy + smooth)
    dynamic_weight2 = -alpha * loss21 - (1 - alpha) * loss20
    dynamic_weight2[dynamic_weight2 > 1] = 1

    dynamic_weight = 0.3 * dynamic_weight1 + 0.7 * dynamic_weight2
    if np.isnan(np.min(dynamic_weight)):
        sdfs = 3
    dynamic_weight = torch.from_numpy(np.array(dynamic_weight))
    dynamic_weight = dynamic_weight.float().cuda()
    return dynamic_weight

def save_data_online(path, image, label, weight, names, limits=1500):
    name_list = os.listdir(os.path.join(path, 'image'))
    name_list.sort(key=lambda x:float(x.split('_')[0]))
    image = image.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    weight = weight.detach().cpu().numpy()
    dice_list = [float(n.split('_')[0]) for n in name_list]
    for i in range(image.shape[0]):
        if len(name_list) < limits:
            np.save(os.path.join(path, 'image', names[i]), image[i])
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))
            np.save(os.path.join(path, 'weight', names[i]), weight[i])
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
        else:
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            if index == 0: continue
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
            os.remove(os.path.join(path, 'image', name_list[0]))
            os.remove(os.path.join(path, 'label', name_list[0]))
            os.remove(os.path.join(path, 'weight', name_list[0]))
            name_list = name_list[1:]
            dice_list = dice_list[1:]
            np.save(os.path.join(path, 'image', names[i]), image[i])
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))
            np.save(os.path.join(path, 'weight', names[i]), weight[i])

def save_data_online3(path, image, label, weight, skel,names, limits=1500):
    name_list = os.listdir(os.path.join(path, 'image'))
    name_list.sort(key=lambda x:float(x.split('_')[0]))
    image = image.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    weight = weight.detach().cpu().numpy()
    skel = skel.detach().cpu().numpy()
    dice_list = [float(n.split('_')[0]) for n in name_list]
    for i in range(image.shape[0]):
        if len(name_list) < limits:
            np.save(os.path.join(path, 'image', names[i]), image[i])
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))
            np.save(os.path.join(path, 'weight', names[i]), weight[i])
            np.save(os.path.join(path, 'skel', names[i]), skel[i].astype(np.int8))
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
        else:
            index = bisect.bisect(dice_list, float(names[i].split('_')[0]))
            if index == 0: continue
            name_list.insert(index, names[i])
            dice_list.insert(index, float(names[i].split('_')[0]))
            os.remove(os.path.join(path, 'image', name_list[0]))
            os.remove(os.path.join(path, 'label', name_list[0]))
            os.remove(os.path.join(path, 'weight', name_list[0]))
            os.remove(os.path.join(path, 'skel', name_list[0]))
            name_list = name_list[1:]
            dice_list = dice_list[1:]
            np.save(os.path.join(path, 'image', names[i]), image[i])
            np.save(os.path.join(path, 'label', names[i]), label[i].astype(np.int8))
            np.save(os.path.join(path, 'weight', names[i]), weight[i])
            np.save(os.path.join(path, 'skel', names[i]), skel[i].astype(np.int8))

def train3(data_root, model_savepath, online_savepath, log_savepath, pred2_path,br_skel_path,BR_weight_path,aug, DTI,
           DWi, start_model, start_epoch, file_path, file_root, gpu):
    max_epoches = 75
    batch_size =8
    aug_flag = aug  #是否数剧增强 否1 是2

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # file_root='./data'

    current_date = datetime.datetime.now()
    log_root = './LOG/'+model_savepath.split('./saved_model/')[-1]+'_{}_{}_{}_{}'.format(current_date.month, current_date.day,
                                                                      current_date.hour,
                                                                      current_date.minute)
    writer = SummaryWriter(str(log_root))

    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    model = DMR_UNet(in_channel=2, n_classes=1)
    train_dataset = AirwayHMData3(file_path=file_path,
                                  data_root=data_root,
                                  file_root=file_root,
                                  pred2_path=pred2_path,
                                  br_skel_path=br_skel_path,
                                  BR_weight_path=BR_weight_path,
                                  batch_size=batch_size,
                                  cube_size=128,
                                  aug_flag=aug_flag)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=10,
                                   pin_memory=True,
                                   drop_last=True)
    valid_dataset = SegValCropData(file_path,
                                   data_root,
                                   batch_size=24,
                                   cube_size=128,
                                   step=64)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=24,
                                  shuffle=False,
                                  num_workers=10,
                                  pin_memory=True,
                                  drop_last=True)

    max_step = len(train_dataset) * max_epoches
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[40, 60],
                                                        gamma=0.1)
    # resume
    # weights_dict = torch.load(os.path.join(model_savepath ,'DMR_UNet_9.pth'))
    weights_dict = torch.load(start_model +
                              'DMR_UNet_{}.pth'.format(start_epoch))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    starttime = time.time()
    for ep in range(max_epoches):
        if os.path.exists(online_savepath):
            shutil.rmtree(online_savepath)
            os.mkdir(online_savepath)
            os.mkdir(online_savepath + '/image')
            os.mkdir(online_savepath + '/label')
            os.mkdir(online_savepath + '/weight')
            os.mkdir(online_savepath + '/skel')
        else:
            os.mkdir(online_savepath)
            os.mkdir(online_savepath + '/image')
            os.mkdir(online_savepath + '/label')
            os.mkdir(online_savepath + '/weight')
            os.mkdir(online_savepath + '/skel')

        for iter, pack in enumerate(train_data_loader):
            data = pack[0].float().cuda()
            data2 = pack[1].float().cuda()
            label = pack[2].float().cuda()
            weight = pack[3].float().cuda()
            skel = pack[4].float().cuda()

            data = data.transpose(0, 1)
            data2 = data2.transpose(0, 1)
            label = label.transpose(0, 1)
            weight = weight.transpose(0, 1)
            skel = skel.transpose(0, 1)

            data = torch.cat([data, data2], dim=1)

            pred_en, pred_de = model(data)
            pred_en = torch.sigmoid(pred_en)
            pred_de = torch.sigmoid(pred_de)

            if DWi == 1:
                dynamic_weight = D_Wi(pred_en, pred_de)
                weight = dynamic_weight + weight

            dice_loss_en = general_union_loss_lib(pred_en, label, weight)
            dice_loss_de = general_union_loss_lib(pred_de, label, weight)
            dice_loss_ori = dice_loss(pred_en, label) + dice_loss(
                pred_de, label)
            break_loss = br_loss(pred_en, label, skel, weight) + br_loss(
                pred_de, label, skel, weight)
            loss = dice_loss_ori * 0.5 + dice_loss_de * 1 + dice_loss_en * 0.5 + break_loss * 0.5  #混合loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            names = [
                general_union_loss_lib(pred_de[i], label[i], weight[i])
                for i in range(data.shape[0])
            ]
            names = [str(n.item()) + '_' + str(iter) + '.npy' for n in names]
            save_data_online3(online_savepath,
                             data,
                             label,
                             weight,
                             skel,
                             names,
                             limits=int(len(train_dataset) * batch_size *
                                        0.3))  #limit of loss???

            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset), '/',
                      max_step, 'loss:', loss.item(), 'dice loss:',
                      dice_loss_ori.item(), 'GUL encode:', dice_loss_en.item(),
                      'GUL decode:', dice_loss_de.item(), 'break loss:',
                      break_loss.item())

            writer.add_scalars('Train', {'loss': loss.item(), 'dice_loss': dice_loss_ori.item(),'GUL_encode': dice_loss_en.item(),'GUL_decode': dice_loss_de.item(),'break_loss': break_loss.item()}, iter + ep * len(train_dataset))
        torch.cuda.empty_cache()

        writer.close()                  
        lr_scheduler.step()
        # '''
        print('start online hard mining: ')
        hm_dataset = OnlineHMData3(data_root=online_savepath,
                                  batch_size=8,
                                  rate=1.0)
        hm_dataloader = DataLoader(dataset=hm_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=2,
                                   pin_memory=True,
                                   drop_last=True)
        for iter, pack in enumerate(hm_dataloader):
            data = pack[0].float().cuda()
            label = pack[1].float().cuda()
            weight = pack[2].float().cuda()
            skel = pack[3].float().cuda()
            pred_en, pred_de = model(data)
            pred_en = torch.sigmoid(pred_en)
            pred_de = torch.sigmoid(pred_de)
            dice_loss_ori = dice_loss(pred_en, label) + dice_loss(
                pred_de, label)
            break_loss = br_loss(pred_en, label, skel, weight) + br_loss(
                pred_de, label, skel, weight)
            dice_loss_all = general_union_loss_lib(pred_en, label, weight) * 0.5 + \
                            general_union_loss_lib(pred_de, label, weight)
            loss = break_loss * 0.5 + dice_loss_all + dice_loss_ori * 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print('epoch:', ep,
                      iter + ep * len(train_dataset), '/', max_step, 'loss:',
                      loss.item(), 'dice loss:', dice_loss_ori.item(), 'GUL:',
                      dice_loss_all.item(), 'break loss:', break_loss.item())

        lr_scheduler.step()
        # '''
        print('')
        validation(data_root, model, valid_dataloader, ep, log_savepath, DTI,
                   file_root)
        print('')

        torch.cuda.empty_cache()
        if not os.path.exists(model_savepath):
            os.mkdir(model_savepath)
        torch.save(
            model.module.state_dict(),
            os.path.join(model_savepath, 'DMR_UNet_' + str(ep) + '.pth'))
        print(ep, ':', time.time() - starttime)
    sdf = 2

def train2(data_root,
           model_savepath,
           online_savepath,
           log_savepath,
           pred_path,
           aug,
           DTI,
           DWi,
           start_model,
           start_epoch,
           file_path,
           file_root,
           gpu='0'):
    max_epoches = 75
    batch_size =8
    aug_flag = aug  #是否数剧增强 否1 是2

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    current_date = datetime.datetime.now()
    log_root = './LOG/'+model_savepath.split('./saved_model/')[-1]+'_{}_{}_{}_{}'.format(current_date.month, current_date.day,
                                                                      current_date.hour,
                                                                      current_date.minute)
    writer = SummaryWriter(str(log_root))


    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    model = DMR_UNet(in_channel=2, n_classes=1)

    train_dataset = AirwayHMData(file_path=file_path,
                                 data_root=data_root,
                                 file_root=file_root,
                                 pred_path=pred_path,
                                 batch_size=batch_size,
                                 cube_size=128,
                                 aug_flag=aug_flag)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=10,
                                   pin_memory=True,
                                   drop_last=True)
    valid_dataset = SegValCropData(file_path,
                                   data_root,
                                   batch_size=24,
                                   cube_size=128,
                                   step=64)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=24,
                                  shuffle=False,
                                  num_workers=10,
                                  pin_memory=True,
                                  drop_last=True)

    max_step = len(train_dataset) * max_epoches
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[40, 60],
                                                        gamma=0.1)

    # resume
    weights_dict = torch.load(os.path.join(model_savepath ,'DMR_UNet_63.pth'))
    # weights_dict = torch.load(start_model +
    #                           'DMR_UNet_{}.pth'.format(start_epoch))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    starttime = time.time()
    for ep in range(64,max_epoches):
        if os.path.exists(online_savepath):
            shutil.rmtree(online_savepath)
            os.mkdir(online_savepath)
            os.mkdir(online_savepath + '/image')
            os.mkdir(online_savepath + '/label')
            os.mkdir(online_savepath + '/weight')
        else:
            os.mkdir(online_savepath)
            os.mkdir(online_savepath + '/image')
            os.mkdir(online_savepath + '/label')
            os.mkdir(online_savepath + '/weight')

        for iter, pack in enumerate(train_data_loader):
            data = pack[0].float().cuda()
            data2 = pack[1].float().cuda()
            label = pack[2].float().cuda()
            weight = pack[3].float().cuda()

            data = data.transpose(0, 1)
            data2 = data2.transpose(0, 1)
            label = label.transpose(0, 1)
            weight = weight.transpose(0, 1)
            data = torch.cat([data, data2], dim=1)

            pred_en, pred_de = model(data)
            pred_en = torch.sigmoid(pred_en)
            pred_de = torch.sigmoid(pred_de)

            if DWi == 1:
                dynamic_weight=D_Wi(pred_en, pred_de)
                weight = dynamic_weight + weight

            dice_loss_en = general_union_loss_lib(pred_en, label, weight)
            dice_loss_de = general_union_loss_lib(pred_de, label, weight)
            dice_loss_ori = dice_loss(pred_en, label) + dice_loss(
                pred_de, label)
            loss = dice_loss_de * 1 + dice_loss_en * 0.5 + dice_loss_ori * 0.5  #混合loss
            if np.isnan(loss.item()):
                sdfs = 3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            names = [
                general_union_loss_lib(pred_de[i], label[i], weight[i])
                for i in range(data.shape[0])
            ]
            names = [str(n.item()) + '_' + str(iter) + '.npy' for n in names]
            save_data_online(online_savepath,
                             data,
                             label,
                             weight,
                             names,
                             limits=int(len(train_dataset) * batch_size *
                                        0.3))  #limit of loss???

            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset), '/',
                      max_step, 'loss:', loss.item(), 'dice loss encode:',
                      dice_loss_en.item(), 'dice loss decode:',
                      dice_loss_de.item(), 'dice loss original:',
                      dice_loss_ori.item())

            writer.add_scalars('Train', {'loss': loss.item(), 'dice_loss_encode': dice_loss_en.item(),'dice_loss_decode': dice_loss_de.item(),'dice_loss_original': dice_loss_ori.item()}, iter + ep * len(train_dataset))
        torch.cuda.empty_cache()

        writer.close()            

        lr_scheduler.step()

        print('start online hard mining: ')
        hm_dataset = OnlineHMData(data_root=online_savepath,
                                  batch_size=8,
                                  rate=1.0)
        hm_dataloader = DataLoader(dataset=hm_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=2,
                                   pin_memory=True,
                                   drop_last=True)
        for iter, pack in enumerate(hm_dataloader):
            data = pack[0].float().cuda()
            label = pack[1].float().cuda()
            weight = pack[2].float().cuda()
            pred_en, pred_de = model(data)
            dice_loss_ori = dice_loss(pred_en, label) + dice_loss(
                pred_de, label)
            pred_en = torch.sigmoid(pred_en)
            pred_de = torch.sigmoid(pred_de)
            dice_loss_all = general_union_loss_lib(pred_en, label, weight) * 0.5 + \
                            general_union_loss_lib(pred_de, label, weight)
            loss = dice_loss_ori * 0.5 + dice_loss_all
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset), '/',
                      max_step, 'loss:', loss.item(), 'dice loss:',
                      dice_loss_all.item(), 'dice loss:', dice_loss_ori.item())
        lr_scheduler.step()

        print('')
        validation(data_root,model, valid_dataloader, ep, log_savepath, DTI, file_root)
        print('')

        torch.cuda.empty_cache()

        if not os.path.exists(model_savepath):
            os.mkdir(model_savepath)
        torch.save(
            model.module.state_dict(),
            os.path.join(model_savepath, 'DMR_UNet_' + str(ep) + '.pth'))
        print(ep, ':', time.time() - starttime)
    sdf = 2

def train(data_root,model_savepath,
          log_savepath,
          aug,
          DTI,
          file_path,
          file_root,
          gpu='0'
          ):
    max_epoches = 100
    batch_size =8
    aug_flag = aug  #是否数剧增强 否1 是2
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    current_date = datetime.datetime.now()
    log_root = './LOG/'+model_savepath.split('./saved_model/')[-1]+'_{}_{}_{}_{}'.format(current_date.month, current_date.day,
                                                                      current_date.hour,
                                                                      current_date.minute)
    writer = SummaryWriter(str(log_root))



    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)

    model = DMR_UNet(in_channel=2, n_classes=1)

    train_dataset = CropSegData(file_path=file_path,
                                data_root=data_root,
                                file_root=file_root,
                                batch_size=batch_size,
                                aug_flag=aug_flag)
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=1,
                                   shuffle=True,
                                   num_workers=10,
                                   pin_memory=True,
                                   drop_last=True)
    valid_dataset = SegValCropData(file_path,
                                   data_root,
                                   batch_size=24,
                                   cube_size=128,
                                   step=64)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=24,
                                  shuffle=False,
                                  num_workers=10,
                                  pin_memory=True,
                                  drop_last=True)

    max_step = len(train_dataset) * max_epoches

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[60, 90],
                                                        gamma=0.1)

    # resume
    # weights_dict = torch.load(os.path.join(model_savepath ,'DMR_UNet_27.pth'))
    # model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    starttime = time.time()
    for ep in range(max_epoches):
        for iter, pack in enumerate(train_data_loader):
            data = pack[0].float().cuda()
            data2 = pack[1].float().cuda()
            label = pack[2].float().cuda()
            weight = pack[3].float().cuda()

            data = data.transpose(
                0, 1)
            data2 = data2.transpose(0, 1)
            label = label.transpose(0, 1)
            weight = weight.transpose(0, 1)
            data = torch.cat([data, data2], dim=1)

            pred_en, pred_de = model(data)
            pred_en = torch.sigmoid(pred_en)
            pred_de = torch.sigmoid(pred_de)
            dice_loss_en = dice_loss(pred_en, label)
            dice_loss_de = dice_loss(pred_de, label)
            loss = dice_loss_de + dice_loss_en

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print('epoch:', ep, iter + ep * len(train_dataset), '/',
                      max_step, 'loss:', loss.item(), 'dice loss encode:',
                      dice_loss_en.item(), 'dice loss decode:',
                      dice_loss_de.item())

            # break
            writer.add_scalars('Train', {'loss': loss.item(), 'dice_loss_encode': dice_loss_en.item(),'dice_loss_decode': dice_loss_de.item()}, iter + ep * len(train_dataset))

        writer.close()
        lr_scheduler.step()

        print('')
        if ep == 99:
            validation(data_root,model, valid_dataloader, ep, log_savepath, DTI,
                       file_root)
        print('')
        torch.cuda.empty_cache()

        if not os.path.exists(model_savepath):
            os.mkdir(model_savepath)
        torch.save(
            model.module.state_dict(),
            os.path.join(model_savepath, 'DMR_UNet_' + str(ep) + '.pth'))
        print(ep, ':', time.time() - starttime)

def validation(data_root,model, valid_dataloader, epoch,log_savepath,DTI,file_root):
    model.train()
    TDs,BDs,DSCs,Pres,Sens,Spes = [], [], [], [],[], []
    last_name = ''
    flag = False
    h_thresh=0.5
    l_thresh=0.35
    with torch.no_grad():
        for i, (x, name, pos) in enumerate(valid_dataloader):
            name = name[0]
            if name != last_name:
                if last_name != '':
                    pred = pred / pred_num
                    pred = np.squeeze(pred)
                    if DTI==0:
                        pred[pred >= 0.5] = 1
                        pred[pred < 0.5] = 0
                    else:
                        pred=double_threshold_iteration(pred,h_thresh,l_thresh)
                    TD,BD,DSC,Pre,Sen,Spe= evaluation_case(pred, label, last_name,file_root)
                    TDs.append(TD)
                    BDs.append(BD)
                    DSCs.append(DSC)
                    Pres.append(Pre)
                    Sens.append(Sen)
                    Spes.append(Spe)

                label = sitk.ReadImage(data_root+'/mask/'+name+ 'mask_cut' + '.nii.gz')
                label = sitk.GetArrayFromImage(label)
                pred = np.zeros(label.shape)
                pred = pred[np.newaxis, np.newaxis, ...]
                pred_num = np.zeros(pred.shape)
                last_name = name

            x = x.cuda()
            p0, p = model(x)
            p = torch.sigmoid(p)
            p = p.cpu().detach().numpy()
            pos = pos.numpy()
            for i in range(len(pos)):
                # print(pos)
                xl, xr, yl, yr, zl, zr = pos[i,0], pos[i,1], pos[i,2], pos[i,3], pos[i,4], pos[i,5]
                pred[0, :, xl:xr, yl:yr, zl:zr] += p[i]
                pred_num[0, :, xl:xr, yl:yr, zl:zr] += 1

        pred = pred / pred_num
        pred = np.squeeze(pred)
        if DTI==0:
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
        else:
            pred=double_threshold_iteration(pred,h_thresh,l_thresh)

        TD,BD,DSC,Pre,Sen,Spe= evaluation_case(pred, label, last_name,file_root)
        TDs.append(TD)
        BDs.append(BD)
        DSCs.append(DSC)
        Pres.append(Pre)
        Sens.append(Sen)
        Spes.append(Spe)

        TD_mean = np.mean(TDs)
        TD_std = np.std(TDs)
        BD_mean = np.mean(BDs)
        BD_std = np.std(BDs)
        DSC_mean = np.mean(DSCs)
        DSC_std = np.std(DSCs)
        Pre_mean = np.mean(Pres)
        Pre_std = np.std(Pres)
        Sen_mean = np.mean(Sens)
        Sen_std = np.std(Sens)
        Spe_mean = np.mean(Spes)
        Spe_std = np.std(Spes)
        print("TD: %0.4f (%0.4f), BD: %0.4f (%0.4f), DSC: %0.4f (%0.4f), Pre: %0.4f (%0.4f), Sen: %0.4f (%0.4f), Spe: %0.4f (%0.4f)" % (
               TD_mean, TD_std,BD_mean,BD_std,DSC_mean,DSC_std,Pre_mean,Pre_std,Sen_mean,Sen_std,Spe_mean,Spe_std))
        line = "TD: %0.4f (%0.4f), BD: %0.4f (%0.4f), DSC: %0.4f (%0.4f), Pre: %0.4f (%0.4f), Sen: %0.4f (%0.4f), Spe: %0.4f (%0.4f)" % (
               TD_mean, TD_std,BD_mean,BD_std,DSC_mean,DSC_std,Pre_mean,Pre_std,Sen_mean,Sen_std,Spe_mean,Spe_std)
        with open(log_savepath, 'a') as file:
            file.writelines(['epoch:' + str(epoch)+'\n', line+'\n', '\n'])

def evaluation_case(pred, label, name,file_root):
    parsing = sitk.ReadImage(os.path.join(file_root, 'tree_parse_val', name + 'mask_cut.nii.gz'))
    parsing = sitk.GetArrayFromImage(parsing)
    # parsing=parsing.transpose(2,1,0)
    if len(pred.shape) > 3:
        pred = pred[0]
    if len(label.shape) > 3:
        label = label[0]

    # cd, num = measure.label(pred, return_num=True, connectivity=2)
    cd=cc3d.connected_components(pred, connectivity=26)
    region = measure.regionprops(cd)
    num_list = [i for i in range(1, np.max(cd)+1)]
    area_list = [region[i-1].area for i in num_list]
    volume_sort = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
    if volume_sort!=[]:
        large_cd =(cd==volume_sort[0]).astype(np.uint8)
    else:
        large_cd=pred.astype(np.uint8)

    skeleton = sitk.ReadImage(os.path.join(file_root, 'skeleton_val', name + 'mask_cut.nii.gz'))
    skeleton = sitk.GetArrayFromImage(skeleton)
    # skeleton=skeleton.transpose(2,1,0)
    skeleton = (skeleton > 0)
    skeleton = skeleton.astype('uint8')

    _,_,BD=branch_detected_calculation(large_cd, parsing, skeleton)
    DSC=dice_coefficient_score_calculation(large_cd, label)
    TD=tree_length_calculation(large_cd, skeleton)
    Sen=sensitivity_calculation(large_cd, label)
    Spe=specificity_calculation(large_cd, label)
    Pre=precision_calculation(large_cd, label)

    print(name, "TD: %0.4f" % (TD),"BD: %0.4f" % (BD), "DSC: %0.4f" % (DSC),"Precision: %0.4f" % (Pre), "Sen: %0.4f" % (Sen),
          "Spe: %0.4f" % (Spe))
    return TD,BD,DSC,Pre,Sen,Spe

def valid_recall(log_path):

    with open(log_path, 'r') as file:
        lines = file.readlines()
    m_lines = []
    for l in range(len(lines)):
        if l % 3 == 1:
            m_lines.append(lines[l])
    TD = []
    BD = []
    DSC = []
    Pre = []
    Sen = []
    SCORE = []
    for l in m_lines:
        TD.append(float(re.findall(r'TD: (.*?) ', l)[0]))
        BD.append(float(re.findall(r'BD: (.*?) ', l)[0]))
        DSC.append(float(re.findall(r'DSC: (.*?) ', l)[0]))
        Pre.append(float(re.findall(r'Pre: (.*?) ', l)[0]))
        Sen.append(float(re.findall(r'Sen: (.*?) ', l)[0]))
    for l in range(len(TD)):
        SCORE.append((TD[l] + BD[l]) * 0.15 + (DSC[l] + Pre[l]) * 0.2 +
                     Sen[l] * 0.3)

    print('第', SCORE.index(max(SCORE)), '个epoch验证效果最佳，分数为', max(SCORE))
    return SCORE.index(max(SCORE))

def valid(log_path):
    with open(log_path, 'r') as file:
        lines=file.readlines()
    m_lines=[]
    for l in range(len(lines)):
        if l%3==1:
            m_lines.append(lines[l])
    TD=[]
    BD=[]
    DSC=[]
    Pre=[]
    SCORE=[]
    for l in m_lines:
        TD.append(float(re.findall(r'TD: (.*?) ', l)[0]))
        BD.append(float(re.findall(r'BD: (.*?) ', l)[0]))
        DSC.append(float(re.findall(r'DSC: (.*?) ', l)[0]))
        Pre.append(float(re.findall(r'Pre: (.*?) ', l)[0]))
    for l in range(len(TD)):
        SCORE.append((TD[l]+BD[l]+DSC[l]+Pre[l])/4)

    print('第',SCORE.index(max(SCORE)),'个epoch验证效果最佳，分数为',max(SCORE))
    return SCORE.index(max(SCORE))

def dtival(model_savepath,log_savepath,data_root,file_path,file_root,dtiep,gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    model = DMR_UNet(in_channel=2, n_classes=1)

    valid_dataset = SegValCropData(file_path, data_root, batch_size=24, cube_size = 128, step = 64)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=24, shuffle=False, num_workers=10,
                                  pin_memory=True, drop_last=True)

    weights_dict = torch.load(os.path.join(model_savepath ,'DMR_UNet_{}.pth'.format(dtiep)))
    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    starttime=time.time()
    print('')
    DTI=1
    validation(data_root,model, valid_dataloader, dtiep,log_savepath,DTI,file_root)
    # print((time.time()-starttime)/60,'mins')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    ###################################### GPU setup
    # gpu_all_num = 8
    # gpu_need_num = 2
    # lm_per_gpu = 40000
    # while 1:
    #     free = np.zeros((gpu_all_num))
    #     for i in range(gpu_all_num):
    #         free[i] = get_gpu_mem_info(i)
    #     if len(np.where(free > lm_per_gpu)[0]) >= gpu_need_num:
    #         break
    # gpu = ','.join(
    #     [str(x) for x in list(np.where(free > lm_per_gpu)[0][0:gpu_need_num])])
    
    gpu = '0,2'

    ###################################### TRAIN_STAGE_1################################################################14 bs
    data_root = '/mnt/yby/CT_DATASET_BAS_90'
    file_path = './data/base_dict.json'
    file_root = './data'
    model_savepath = './saved_model/stage_one'
    log_savepath = './LOG/log_stage_one.txt'
    aug = 2  #1=不
    DTI = 1  #0=不
    # train(data_root, model_savepath, log_savepath, aug, DTI, file_path,
    #       file_root, gpu)

    ###################################### STAGE_1_PRED
    pred1_path = './data/pred_1'
    # save_gradients_tw(data_root,
    #                   stage_load=model_savepath,
    #                   file_root=file_root,
    #                   savepath=pred1_path,
    #                   file_path=file_path,
    #                   layer=0,
    #                   gpu=gpu)

    ###################################### TRAIN_STAGE_2
    start_model = './saved_model/stage_one/'
    start_epoch = 99

    data_root = '/mnt/yby/CT_DATASET_BAS_90'
    model_savepath = './saved_model/stage_two'
    online_savepath = './data/online_hardmining_stage_two'
    log_savepath = './LOG/log_stage_two.txt'
    aug = 2  #1=不
    DTI = 0
    DWi = 1  #0=不
    train2(data_root, model_savepath, online_savepath, log_savepath,pred1_path, aug, DTI,
           DWi, start_model, start_epoch, file_path, file_root, gpu)

    ###################################### STAGE_2_PRED/BR-wi/br_skel
    epoch = valid_recall(log_savepath)
    whichepoch = './saved_model/stage_two/DMR_UNet_{}.pth'.format(epoch)
    pred2_path = './data/pred_2'
    save_s2_pred(data_root, whichepoch, pred2_path, file_root, file_path,gpu)

    BR_weight_path = './data/BR_weight'
    br_skel_path = './data/br_skel'
    save_weight_break(data_root, pred2_path, BR_weight_path, br_skel_path, file_root,file_path)

    ###################################### TRAIN_STAGE_3
    start_model = './saved_model/stage_two/'
    log_path = './LOG/log_stage_two.txt'
    best_epoch = valid_recall(log_path)
    start_epoch = best_epoch

    model_savepath = './saved_model/stage_three'
    online_savepath = './data/online_hardmining_stage_three'
    log_savepath = './LOG/log_stage_three.txt'
    aug = 2  #1=不
    DTI = 0
    DWi = 1  #0=不
    train3(data_root,model_savepath, online_savepath, log_savepath,pred2_path,br_skel_path,BR_weight_path,aug, DTI, DWi,
           start_model, start_epoch, file_path, file_root, gpu)
    
    ###### DTI结果
    best_epoch = valid_recall( './LOG/log_stage_two.txt')
    dtival( './saved_model/stage_two', './LOG/log_stage_two.txt',data_root,file_path,file_root,best_epoch,gpu)

    best_epoch = valid( './LOG/log_stage_three.txt')
    dtival('./saved_model/stage_three', './LOG/log_stage_three.txt',data_root,file_path,file_root,best_epoch,gpu)

