# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 19:37:26 2021

@author: Hao Zheng
"""

import numpy as np
import torch
import os
import nibabel
import SimpleITK as sitk
from SE_UNet import DMR_UNet
from data import load_json_file

torch.manual_seed(777) # cpu
torch.cuda.manual_seed(777) #gpu
np.random.seed(777) #numpy

def dice_loss(pred, target):
    smooth = 1.
    iflat = pred
    tflat = target
    intersection = ((iflat) * tflat).sum()   
    return 1-((2. * intersection + smooth)/((iflat).sum() + (tflat).sum() + smooth))

def Tversky_loss(pred, target):
    smooth = 1.0
    alpha = 0.05
    beta = 1-alpha
    intersection = (pred*target).sum()
    FP = (pred*(1-target)).sum()
    FN = ((1-pred)*target).sum()
    return 1-(intersection + smooth)/(intersection + alpha*FP + beta*FN + smooth)

def root_Tversky_loss(pred, target, dist):
    alpha0 = 1
    beta0 = 1
    alpha = 0.05
    beta = 1 - alpha
    weight = (0.95*dist+0.05)*alpha0*target + beta0*(1-target)*dist
    #weight = 1
    smooth = 1.0
    sigma1 = 0.0001
    sigma2 = 0.0001
    weight_i = target*sigma1 + (1-target)*sigma2
    intersection = (weight*((pred+weight_i)**0.7)*target).sum()
    intersection2 = (weight*(alpha*pred + beta*target)).sum()
    return 1-(intersection + smooth)/(intersection2 + smooth)


def process_img(data):
    data = data.astype(np.float32)
    data2 = data.copy()
    data2[data2 > 500] = 500
    data2[data2 < -1000] = -1000
    data2 = (data2 + 1000) / 1500
    data[data > 1024] = 1024
    data[data < -1024] = -1024
    data = (data + 1024) / 2048
    return data, data2

def save_gradients_tw(data_root,stage_load,file_root,savepath,file_path,layer=0,gpu='0'):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    case_net = DMR_UNet(in_channel=2, n_classes=1)
    weights_dict = torch.load(os.path.join(stage_load, 'DMR_UNet_99.pth'))
    case_net.load_state_dict(weights_dict, strict=False)
    case_net = torch.nn.DataParallel(case_net).cuda()
    case_net.cuda()
    case_net.train()

    # load data
    file_list = load_json_file(file_path, folder='0', mode=['train', 'val'])
    file_list.sort()
    for idx in range(len(file_list)):
        name = file_list[idx]
        img = sitk.ReadImage(data_root+'/data/'+name+ 'data_cut' + '.nii.gz')
        img = sitk.GetArrayFromImage(img)
        label = sitk.ReadImage(data_root+'/mask/'+name+ 'mask_cut' + '.nii.gz')
        label = sitk.GetArrayFromImage(label)
        img, img2 = process_img(img)
        weight = np.load(os.path.join(file_root+'/LIB_weight', name+'.npy'))

        # calculate gradients
        img = img[np.newaxis, np.newaxis, ...]
        img2 = img2[np.newaxis, np.newaxis, ...]
        label = label[np.newaxis, np.newaxis, ...]
        weight = weight[np.newaxis, np.newaxis, ...]
        x = torch.from_numpy(img.astype(np.float32)).cuda()
        x2 = torch.from_numpy(img2.astype(np.float32)).cuda()
        x = torch.cat((x, x2), dim=1)
        y = torch.from_numpy(label.astype(np.float32)).cuda()
        w = torch.from_numpy(weight.astype(np.float32)).cuda()

        cube_size = 128
        step = 64
        pred = np.zeros(y.shape)
        pred_num = np.zeros(y.shape)
        grads = np.zeros(y.shape)
        grads_num = np.zeros(y.shape)
        # sliding window
        xnum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 \
            else (x.shape[ 2] - cube_size) // step + 2
        ynum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 \
            else (x.shape[3] - cube_size) // step + 2
        znum = (x.shape[4] - cube_size) // step + 1 if (x.shape[4] - cube_size) % step == 0 \
            else (x.shape[4] - cube_size) // step + 2
        for xx in range(xnum):
            xl = step * xx
            xr = step * xx + cube_size
            if xr > x.shape[2]:
                xr = x.shape[2]
                xl = x.shape[2] - cube_size
            for yy in range(ynum):
                yl = step * yy
                yr = step * yy + cube_size
                if yr > x.shape[3]:
                    yr = x.shape[3]
                    yl = x.shape[3] - cube_size
                for zz in range(znum):
                    zl = step * zz
                    zr = step * zz + cube_size
                    if zr > x.shape[4]:
                        zr = x.shape[4]
                        zl = x.shape[4] - cube_size

                    x_input = x[:, :, xl:xr, yl:yr, zl:zr]
                    p0, p = case_net(x_input)
                    p_numpy = p.cpu().detach().numpy()
                    pred[:, :, xl:xr, yl:yr, zl:zr] += p_numpy
                    pred_num[:, :, xl:xr, yl:yr, zl:zr] += 1


        pred = pred / pred_num
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        print(name, 'dice', 2 * (pred * label).sum() / (pred + label).sum())
        # np.save(os.path.join('./data/LIBBP/grads', name+'.npy'), grads[0,0])
        pred_nii = nibabel.Nifti1Image(pred[0].astype(np.uint8), np.eye(4))
        nibabel.save(pred_nii, os.path.join(os.path.join(savepath, name+'.nii.gz')))




if __name__ == '__main__':
    whichmodel=1#0=old
    # save_gradients_tw(whichmodel,layer=0)















