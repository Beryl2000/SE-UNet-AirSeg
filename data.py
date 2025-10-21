import numpy as np
import torch
import os
import random
import json
import SimpleITK as sitk
import scipy.ndimage as ndimage
import nibabel
from torch.utils.data import Dataset
from copy import deepcopy

np.random.seed(777)  # numpy

def load_json_file(file_path, folder='0', mode=['train']):
    with open(file_path, 'r') as file:
        data = json.load(file)
    file_list = []
    if int(folder) >= 0:
        for m in mode:
            file_list += data[folder][m]
    else:
        file_list = data[mode[0]]
    file_list = [f.split('.')[0] for f in file_list]
    return file_list


# def npy2niigz():
#     npy_root = './data/pred'
#     niigz_root = './data/pred2'
#     file_list = os.listdir(npy_root)
#     file_list.sort()
#     for f in file_list:
#         name = f.split('.')[0]
#         pred = np.load(os.path.join(npy_root, f))
#         pred = (pred > 0.5).astype(int)
#         pred_nii = nibabel.Nifti1Image(pred.astype(np.uint8), np.eye(4))
#         nibabel.save(pred_nii, os.path.join(os.path.join('./data/pred2', name + '.nii.gz')))
#         print(name, pred.shape)

def random_flip(data_list):
    result=deepcopy(data_list)
    flipid = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    while (flipid  == [1,  1, 1]).all():
        flipid = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    for i in range(len(result)):
        result[i] = np.ascontiguousarray(result[i][::flipid[0], ::flipid[1], ::flipid[2]])
    return result

def random_rotate(data_list):
    def rotate_left(data):
        data = data.transpose((0, 2, 1))
        data = np.ascontiguousarray(data[:, ::-1])
        return data
    def rotate_right(data):
        data = np.ascontiguousarray(data[:, ::-1])
        data = data.transpose((0, 2, 1))
        data = np.ascontiguousarray(data[:, ::-1])
        return data

    result=deepcopy(data_list)
    k=random.random()
    for i in range(len(result)):
        if  k> 0.5:
            result[i] = rotate_left(result[i])
        else:
            result[i] = rotate_right(result[i])
    return result

def random_color(data, rate=0.2):
    r1 = (random.random() - 0.5) * 2 * rate
    r2 = (random.random() - 0.5) * 2 * rate
    data = data * (1 + r2) + r1
    return data

def central_crop(sample, label, dist, crop_size):
    origin_size = sample.shape
    crop_size = np.array(crop_size)
    start = (origin_size - crop_size) // 2
    sample = sample[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
             start[2]:(start[2] + crop_size[2])]
    label = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1], start[2]:start[2] + crop_size[2]]
    dist = dist[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1], start[2]:start[2] + crop_size[2]]
    return sample, label, dist

def skeleton_sample(img, label, weight, loc, cube_size):
    origin_size = img.shape
    crop_size = np.array([cube_size, cube_size, cube_size])
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop

def small_airway_sample(img, label, weight, loc, cube_size):
    origin_size = img.shape
    crop_size = [cube_size, cube_size, cube_size]
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    return img_crop, label_crop, weight_crop

def hard_sample(img, label, weight, loc_skeleton, loc_small, cube_size):
    origin_size = img.shape
    crop_size = np.array([cube_size, cube_size, cube_size])

    if np.random.random() > 0.5 and len(loc_skeleton[0]) > 0:
        loc = loc_skeleton
    elif len(loc_small[0]) > 0:
        loc = loc_small
    else:
        x = np.random.randint(0, origin_size[0] - cube_size)
        y = np.random.randint(0, origin_size[1] - cube_size)
        z = np.random.randint(0, origin_size[2] - cube_size)
        return (
            img[x:x+cube_size, y:y+cube_size, z:z+cube_size],
            label[x:x+cube_size, y:y+cube_size, z:z+cube_size],
            weight[x:x+cube_size, y:y+cube_size, z:z+cube_size],
        )

    random_loc = np.random.randint(len(loc[0]))
    start = [
        np.random.randint(max(0, loc[0][random_loc] - cube_size // 2), loc[0][random_loc] + cube_size // 2),
        np.random.randint(max(0, loc[1][random_loc] - cube_size // 2), loc[1][random_loc] + cube_size // 2),
        np.random.randint(max(0, loc[2][random_loc] - cube_size // 2), loc[2][random_loc] + cube_size // 2)
    ]

    for i in range(3):
        if (start[i] + cube_size) > origin_size[i]:
            start[i] = origin_size[i] - cube_size

    img_crop = img[start[0]:start[0]+cube_size, start[1]:start[1]+cube_size, start[2]:start[2]+cube_size]
    label_crop = label[start[0]:start[0]+cube_size, start[1]:start[1]+cube_size, start[2]:start[2]+cube_size]
    weight_crop = weight[start[0]:start[0]+cube_size, start[1]:start[1]+cube_size, start[2]:start[2]+cube_size]

    return img_crop, label_crop, weight_crop

def random_sample(img, label, weight, cube_size):
    origin_size = img.shape
    crop_size = [cube_size, cube_size, cube_size]
    start = [np.random.randint(0, origin_size[0] - crop_size[0]), np.random.randint(0, origin_size[1] - crop_size[1]),
             np.random.randint(0, origin_size[2] - crop_size[2])]
    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop

def skeleton_sample_wg(img, label, weight,skeleton, loc, cube_size):
    origin_size = img.shape
    crop_size = np.array([cube_size, cube_size, cube_size])
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skel_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop,skel_crop

def break_sample_wg(img, label, weight,skeleton, loc, cube_size):
    origin_size = img.shape
    crop_size = np.array([cube_size, cube_size, cube_size])
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skel_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop,skel_crop

def small_airway_sample_wg(img, label, weight,skeleton, loc, cube_size):
    origin_size = img.shape
    crop_size = [cube_size, cube_size, cube_size]
    random_loc = np.random.randint(len(loc[0]))
    start = [np.random.randint(max(0, loc[0][random_loc] - crop_size[0] // 2), loc[0][random_loc] + crop_size[0] // 2),
             np.random.randint(max(0, loc[1][random_loc] - crop_size[1] // 2), loc[1][random_loc] + crop_size[1] // 2),
             np.random.randint(max(0, loc[2][random_loc] - crop_size[2] // 2), loc[2][random_loc] + crop_size[2] // 2)]
    for i in range(3):
        if (start[i] + crop_size[i]) > origin_size[i]:
            start[i] = origin_size[i] - crop_size[i]

    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skel_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]

    return img_crop, label_crop, weight_crop,skel_crop

def random_sample_wg(img, label, weight,skeleton, cube_size):
    origin_size = img.shape
    crop_size = [cube_size, cube_size, cube_size]
    start = [np.random.randint(0, origin_size[0] - crop_size[0]), np.random.randint(0, origin_size[1] - crop_size[1]),
             np.random.randint(0, origin_size[2] - crop_size[2])]
    img_crop = img[start[0]:(start[0] + crop_size[0]), start[1]:(start[1] + crop_size[1]),
                   start[2]:(start[2] + crop_size[2])]
    label_crop = label[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                       start[2]:start[2] + crop_size[2]]
    weight_crop = weight[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    skel_crop = skeleton[start[0]:start[0] + crop_size[0], start[1]:start[1] + crop_size[1],
                         start[2]:start[2] + crop_size[2]]
    return img_crop, label_crop, weight_crop,skel_crop

class AirwayHMData(Dataset):

    def __init__(self,
                 file_path,
                 data_root,
                 file_root,
                 pred_path,
                 batch_size,
                 cube_size,
                 aug_flag=1):
        self.data_root = data_root
        self.file_root = file_root
        self.pred_path=pred_path
        self.file_list = load_json_file(file_path, folder='0', mode=['train'])
        self.batch_size = batch_size
        self.cube_size = cube_size
        self.aug_flag = aug_flag


        # Random : Hard = 7 : 3
        self.random_ratio = 0.7
        self.hard_ratio = 0.3

        # scheduler 
        self.decay_step = 5          # 每 5 epoch 调整一次
        self.decay_rate = 0.05         # 每次 Hard 比例增加 10%
        self.max_hard_ratio = 0.8     # 上限
        self.min_hard_ratio = 0.2     # 下限

    def __len__(self):
        return len(self.file_list)

    def process_img(self, img_crops):
        img2_crops = []
        for i in range(len(img_crops)):
            crop = img_crops[i]
            crop2 = crop.copy()
            crop2[crop2 > 500] = 500
            crop2[crop2 < -1000] = -1000
            crop2 = (crop2 + 1000) / 1500
            crop[crop > 1024] = 1024
            crop[crop < -1024] = -1024
            crop = (crop + 1024) / 2048
            img2_crops.append(crop2)  #
            img_crops[i] = crop
        return img_crops, img2_crops

    def crop(self, img, label, weight, pred, skeleton, parsing):
        img_crops, label_crops, weight_crops = [], [], []

        dis = ndimage.distance_transform_edt(label)
        loc_small = np.where((dis * skeleton) < 2)
        loc_skeleton = np.where(skeleton * (1 - pred))

        cube_size = self.cube_size

        # -------- 核心采样循环 --------
        for _ in range(self.batch_size):
            # 按当前动态比例选择
            if np.random.random() < self.hard_ratio:
                # Hard mining（50% skeleton / 50% small airway）
                img_crop, label_crop, weight_crop = hard_sample(
                    img, label, weight, loc_skeleton, loc_small, cube_size
                )
            else:
                # Random crop
                img_crop, label_crop, weight_crop = random_sample(img, label, weight, cube_size)

            img_crops.append(img_crop)
            label_crops.append(label_crop)
            weight_crops.append(weight_crop)

        return img_crops, label_crops, weight_crops


    def update_scheduler(self, epoch, val_loss_random_list, val_loss_hard_list, val_td_list, val_bd_list):
        if epoch % self.decay_step == 0 and epoch != 0:
            window = min(3, len(val_loss_random_list))
            avg_random = np.mean(val_loss_random_list[-window:])
            avg_hard = np.mean(val_loss_hard_list[-window:])
            diff = avg_random - avg_hard

            if len(val_td_list) > 1:
                td_trend = val_td_list[-1] - val_td_list[-2]
                bd_trend = val_bd_list[-1] - val_bd_list[-2]
            else:
                td_trend = bd_trend = 0.0

            print(f"[Scheduler] Epoch {epoch}: diff={diff:.4f}, TD_trend={td_trend:.4f}, BD_trend={bd_trend:.4f}")

            if diff > 0.04 or td_trend < 0 or bd_trend < 0:
                self.hard_ratio = min(self.max_hard_ratio, self.hard_ratio + self.decay_rate)
            elif diff < 0.02 and td_trend >= 0 and bd_trend >= 0:
                self.hard_ratio = max(self.min_hard_ratio, self.hard_ratio - self.decay_rate)
            elif diff > 0.05 or td_trend < -1 or bd_trend < -1:
                self.hard_ratio = max(self.min_hard_ratio, self.hard_ratio - self.decay_rate)

            print(f"[Scheduler] Updated hard_ratio -> {self.hard_ratio:.2f}")

    def augment(self, data_list):
        if random.random() > 0.5:
            result = random_flip(data_list)
            data_list = result
        if random.random() > 0.5:
            result = random_rotate(data_list)
            data_list = result
        return data_list

    def __getitem__(self, item):
        name = self.file_list[item]
        img = sitk.ReadImage(self.data_root + '/data/' + name + 'data_cut' +
                             '.nii.gz')
        img = sitk.GetArrayFromImage(img)
        img=img-1024
        label = sitk.ReadImage(self.data_root + '/mask/' + name + 'mask_cut' +
                               '.nii.gz')
        label = sitk.GetArrayFromImage(label)

        weight = np.load(
            os.path.join(self.file_root, 'LIB_weight', name + '.npy'))
        pred = nibabel.load(
            os.path.join(self.pred_path, name + '.nii.gz'))
        pred = pred.get_fdata()[0]

        parsing = sitk.ReadImage(
            os.path.join(self.file_root, 'tree_parse',
                         name + 'mask_cut.nii.gz'))
        parsing = sitk.GetArrayFromImage(parsing)
        # parsing = parsing.transpose(2, 1, 0)
        skeleton = sitk.ReadImage(
            os.path.join(self.file_root, 'skeleton', name + 'mask_cut.nii.gz'))
        skeleton = sitk.GetArrayFromImage(skeleton)
        # skeleton = skeleton.transpose(2, 1, 0)

        weight = weight**(np.random.random() + 2) * label + (1 - label)
        img_crops, label_crops, weight_crops = self.crop(
            img, label, weight, pred, skeleton, parsing)
        img_crops, img2_crops = self.process_img(img_crops)
        data_list = [img_crops, img2_crops, label_crops, weight_crops]
        if self.aug_flag == 1:
            for i in range(len(data_list[0])):
                _aug = self.augment(
                    [data_list[j][i] for j in range(len(data_list))])
                for j in range(len(data_list)):
                    data_list[j][i] = _aug[j]
        img_crops, img2_crops, label_crops, weight_crops = data_list[
            0], data_list[1], data_list[2], data_list[3]

        img_crops = torch.from_numpy(np.array(img_crops))
        img2_crops = torch.from_numpy(np.array(img2_crops))
        label_crops = torch.from_numpy(np.array(label_crops))
        weight_crops = torch.from_numpy(np.array(weight_crops))

        # if img_crops.shape!=torch.Size([10, 128, 128, 128]):
        #     dsfs=3

        return img_crops, img2_crops, label_crops, weight_crops

class AirwayHMData3(Dataset):
    def __init__(self, file_path, data_root, file_root,pred2_path,br_skel_path,BR_weight_path,batch_size, cube_size,aug_flag=1):
        self.data_root = data_root
        self.file_root = file_root
        self.pred2_path=pred2_path
        self.br_skel_path=br_skel_path
        self.BR_weight_path=BR_weight_path
        self.file_list = load_json_file(file_path, folder='0', mode=['train'])
        self.batch_size = batch_size
        self.cube_size = cube_size
        self.aug_flag = aug_flag

        self.hard_ratio = 0.8       # Hard + Break 总体比例
        self.break_ratio = 0.625    # Hard 内 Break占比（0.5 / 0.8）
        self.min_hard_ratio = 0.2
        self.max_hard_ratio = 0.8
        self.min_break_ratio = 0.2
        self.max_break_ratio = 0.8
        self.decay_rate = 0.05
        self.decay_step = 5        # 每10 epoch更新一次

    def __len__(self):
        return len(self.file_list)

    def process_img(self, img_crops):
        img2_crops = []
        for i in range(len(img_crops)):
            crop = img_crops[i]
            crop2 = crop.copy()
            crop2[crop2 > 500] = 500
            crop2[crop2 < -1000] = -1000
            crop2 = (crop2 + 1000) / 1500
            crop[crop > 1024] = 1024
            crop[crop < -1024] = -1024
            crop = (crop + 1024) / 2048
            img2_crops.append(crop2) 
            img_crops[i] = crop
        return img_crops, img2_crops

    def crop(self, img, label, weight, pred, skeleton, parsing, br_skel):
        """
        Stage 3 crop: Dynamic sampling including Random, Hard, and Break samples.
        """
        img_crops, label_crops, weight_crops, skel_crops = [], [], [], []

        dis = ndimage.distance_transform_edt(label)
        loc_small = np.where((dis * skeleton) < 2)
        loc_skeleton = np.where(skeleton * (1 - pred))
        loc_break = br_skel  # 已经是 break skeleton 的 mask
        cube_size = self.cube_size

        for _ in range(self.batch_size):
            # ----------------- Random vs Hard+Break -----------------
            if np.random.random() < self.hard_ratio:
                # ----------------- Hard mining -----------------
                if np.random.random() < self.break_ratio and len(loc_break[0]) != 0:
                    # Break sample
                    img_crop, label_crop, weight_crop, skel_crop = break_sample_wg(
                        img, label, weight, skeleton, loc_break, cube_size
                    )
                else:
                    # Skeleton 或 Small airway
                    if np.random.random() < 0.5:
                        img_crop, label_crop, weight_crop, skel_crop = small_airway_sample_wg(
                            img, label, weight, skeleton, loc_small, cube_size
                        )
                    else:
                        img_crop, label_crop, weight_crop, skel_crop = skeleton_sample_wg(
                            img, label, weight, skeleton, loc_skeleton, cube_size
                        )
            else:
                # ----------------- Random crop -----------------
                img_crop, label_crop, weight_crop, skel_crop = random_sample_wg(
                    img, label, weight, skeleton, cube_size
                )

            img_crops.append(img_crop)
            label_crops.append(label_crop)
            weight_crops.append(weight_crop)
            skel_crops.append(skel_crop)

        return img_crops, label_crops, weight_crops, skel_crops

    def update_scheduler(self, epoch, val_loss_random_list, val_loss_hard_list, val_td_list, val_bd_list):
        """
        Adaptive adjustment of hard_ratio and break_ratio according to validation metrics.
        """
        if epoch % self.decay_step != 0 or epoch == 0:
            return  # only update every decay_step

        # ------------------ Compute average losses ------------------
        window = min(3, len(val_loss_random_list))
        avg_random = np.mean(val_loss_random_list[-window:])
        avg_hard = np.mean(val_loss_hard_list[-window:])
        diff = avg_random - avg_hard

        # Compute TD/BD trends
        if len(val_td_list) > 1:
            td_trend = val_td_list[-1] - val_td_list[-2]
            bd_trend = val_bd_list[-1] - val_bd_list[-2]
        else:
            td_trend = bd_trend = 0.0

        print(f"[Scheduler] Epoch {epoch}: diff={diff:.4f}, TD_trend={td_trend:.4f}, BD_trend={bd_trend:.4f}")

        # ------------------ Adjust hard_ratio ------------------
        step = self.decay_rate
        if diff > 0.04 or td_trend < 0 or bd_trend < 0:
            # Hard examples are worse or tree quality decreased → increase Hard ratio
            self.hard_ratio = min(self.max_hard_ratio, self.hard_ratio + step)
        elif diff < 0.02 and td_trend >= 0 and bd_trend >= 0:
            # Hard close to overall and tree improving → decrease Hard ratio
            self.hard_ratio = max(self.min_hard_ratio, self.hard_ratio - step)
        # else: keep unchanged

        # ------------------ Adjust break_ratio within Hard ------------------
        # Simple rule: if TD/BD trend < 0, increase break_ratio to focus on break points
        if td_trend < 0 or bd_trend < 0:
            self.break_ratio = min(self.max_break_ratio, self.break_ratio + step)
        elif td_trend > 0 and bd_trend > 0:
            self.break_ratio = max(self.min_break_ratio, self.break_ratio - step)
        # else: keep unchanged

        print(f"[Scheduler] Updated hard_ratio -> {self.hard_ratio:.2f}, break_ratio -> {self.break_ratio:.2f}")

    def augment(self, data_list):
        if random.random() > 0.5:
            result = random_flip(data_list)
            data_list = result
        if random.random() > 0.5:
            result = random_rotate(data_list)
            data_list = result
        return data_list

    def __getitem__(self, item):
        name = self.file_list[item]
        img = sitk.ReadImage(self.data_root+'/data/'+name+ 'data_cut' + '.nii.gz')
        img = sitk.GetArrayFromImage(img)
        img=img-1024
        label = sitk.ReadImage(self.data_root+'/mask/'+name+ 'mask_cut' + '.nii.gz')
        label = sitk.GetArrayFromImage(label)
        r=0.6
        weight = np.load(os.path.join(self.file_root, 'LIB_weight', name + '.npy'))
        br_weight = np.load(os.path.join(self.BR_weight_path, name + '.npy'))

        weight = weight +r* br_weight
        br_skel = np.load(os.path.join(self.br_skel_path, name + '.npy'))
        pred = nibabel.load(os.path.join(self.pred2_path, name + '.nii.gz'))
        pred = pred.get_fdata()[0]

        parsing = sitk.ReadImage(os.path.join(self.file_root, 'tree_parse', name + 'mask_cut.nii.gz'))
        parsing = sitk.GetArrayFromImage(parsing)
        # parsing=parsing.transpose(2,1,0)
        skeleton = sitk.ReadImage(os.path.join(self.file_root, 'skeleton', name + 'mask_cut.nii.gz'))
        skeleton = sitk.GetArrayFromImage(skeleton)
        # skeleton=skeleton.transpose(2,1,0)


        weight = weight ** (np.random.random() + 2) * label + (1 - label)
        img_crops, label_crops, weight_crops,skel_crops = self.crop(img, label, weight, pred, skeleton, parsing,br_skel)
        img_crops, img2_crops = self.process_img(img_crops)
        data_list = [img_crops, img2_crops, label_crops, weight_crops,skel_crops]
        if self.aug_flag==1:
            for i in range(len(data_list[0])):
                _aug = self.augment([data_list[j][i] for j in range(len(data_list))])
                for j in range(len(data_list)):
                    data_list[j][i] = _aug[j]
        img_crops, img2_crops, label_crops, weight_crops,skel_crops  = data_list[0], data_list[1], data_list[2], data_list[3],data_list[4]
        img_crops = torch.from_numpy(np.array(img_crops))
        img2_crops = torch.from_numpy(np.array(img2_crops))
        label_crops = torch.from_numpy(np.array(label_crops))
        weight_crops = torch.from_numpy(np.array(weight_crops))
        skel_crops = torch.from_numpy(np.array(skel_crops))

        return img_crops, img2_crops, label_crops, weight_crops,skel_crops

class OnlineHMData(Dataset):
    def __init__(self, data_root, batch_size, rate=0.33):
        self.data_root = data_root
        self.batch_size = batch_size
        self.name_list = os.listdir(os.path.join(data_root, 'image'))
        self.name_list.sort(key=lambda x:float(x.split('_')[0]))
        self.name_list = self.name_list[-int(rate * len(self.name_list)):]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        image = np.load(os.path.join(self.data_root, 'image', name))
        label = np.load(os.path.join(self.data_root, 'label', name))
        weight = np.load(os.path.join(self.data_root, 'weight', name))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)

        return image, label, weight

class OnlineHMData3(Dataset):
    def __init__(self, data_root, batch_size, rate=0.33):
        self.data_root = data_root
        self.batch_size = batch_size
        self.name_list = os.listdir(os.path.join(data_root, 'image'))
        self.name_list.sort(key=lambda x:float(x.split('_')[0]))
        self.name_list = self.name_list[-int(rate * len(self.name_list)):]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        name = self.name_list[item]
        image = np.load(os.path.join(self.data_root, 'image', name))
        label = np.load(os.path.join(self.data_root, 'label', name))
        weight = np.load(os.path.join(self.data_root, 'weight', name))
        skel = np.load(os.path.join(self.data_root, 'skel', name))
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        weight = torch.from_numpy(weight)
        skel = torch.from_numpy(skel)

        return image, label, weight,skel
    
class CropSegData(Dataset):
    def __init__(self, file_path, data_root,file_root, batch_size, aug_flag,ifgrad=False):
        super(CropSegData, self).__init__()
        self.file_list = load_json_file(file_path)
        self.data_root = data_root
        self.file_root = file_root
        self.batch_size = batch_size
        self.aug_flag = aug_flag
        self.ifgrad = ifgrad

    def __len__(self):
        return len(self.file_list)

    def crop(self, data_list, crop_size=[128, 128, 128]):
        _data = data_list[0]
        shape = _data.shape
        random_range = [[crop_size[i]//2, shape[i]-crop_size[i]//2] for i in range(3)]
        random_center = []
        for i in range(self.batch_size):
            z = random.randint(random_range[0][0], random_range[0][1])
            y = random.randint(random_range[1][0], random_range[1][1])
            x = random.randint(random_range[2][0], random_range[2][1])
            random_center.append([z, y ,x])
        out_list = []
        for data in data_list:
            out = []
            for c in random_center:
                z, y, x = c[0], c[1], c[2]
                out.append(data[z-crop_size[0]//2 : z+crop_size[0]//2,
                                y-crop_size[1]//2 : y+crop_size[1]//2,
                                x-crop_size[2]//2 : x+crop_size[2]//2])
            out_list.append(out)
        return out_list

    def process_imgmsk(self, data, mask):
        data = data.astype(np.float32)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048
        mask = (mask > 0).astype(np.int32).astype(np.float32)


        return data, data2, mask

    def augment(self, data_list):
        if random.random() > 0.5:
            result = random_flip(data_list)
            data_list = result
        if random.random() > 0.5:
            result = random_rotate(data_list)
            data_list = result
        return data_list



    def __getitem__(self, item):
        name = self.file_list[item]
        img = sitk.ReadImage(self.data_root+'/data/'+name+ 'data_cut' + '.nii.gz')
        img = sitk.GetArrayFromImage(img)
        img=img-1024
        label = sitk.ReadImage(self.data_root+'/mask/'+name+ 'mask_cut' + '.nii.gz')
        label = sitk.GetArrayFromImage(label)
        weight = np.load(os.path.join(self.file_root, 'LIB_weight', name+'.npy'))

        img, img2, label = self.process_imgmsk(img, label)
        weight = weight ** (np.random.random() + 2) * label + (1 - label)
        data_list = self.crop([img, img2, label, weight])
        if self.aug_flag==1:

            for i in range(len(data_list[0])):
                _aug = self.augment([data_list[j][i] for j in range(len(data_list))])
                for j in range(len(data_list)):
                    data_list[j][i] = _aug[j]
        img, img2, label, weight = data_list[0], data_list[1], data_list[2], data_list[3]
        img = torch.from_numpy(np.array(img))
        img2 = torch.from_numpy(np.array(img2))
        label = torch.from_numpy(np.array(label))
        weight = torch.from_numpy(np.array(weight))

        return img, img2, label, weight, name

class SegValCropData(Dataset):
    def __init__(self, file_path, data_root, batch_size, cube_size=128, step=64):
        self.file_list = load_json_file(file_path, folder='0', mode=['val'])
        self.root = data_root
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.step = step
        self.file_dic, self.pos_list = self.crop_pos()
        self.last_name = ''
        self.img = None

    def __len__(self):
        return len(self.file_list)

    def crop_pos(self):
        file_dic = {}
        cube_size, step = self.cube_size, self.step
        for f in self.file_list:
            tmp = []
            img = sitk.ReadImage(self.root+'/data/'+f+ 'data_cut' + '.nii.gz')
            x = sitk.GetArrayFromImage(img)
            x = x[np.newaxis, ...]
            xnum = (x.shape[1] - cube_size) // step + 1 if (x.shape[1] - cube_size) % step == 0 else \
                (x.shape[1] - cube_size) // step + 2
            ynum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 else \
                (x.shape[2] - cube_size) // step + 2
            znum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 else \
                (x.shape[3] - cube_size) // step + 2
            for xx in range(xnum):
                xl = step * xx
                xr = step * xx + cube_size
                if xr > x.shape[1]:
                    xr = x.shape[1]
                    xl = x.shape[1] - cube_size
                for yy in range(ynum):
                    yl = step * yy
                    yr = step * yy + cube_size
                    if yr > x.shape[2]:
                        yr = x.shape[2]
                        yl = x.shape[2] - cube_size
                    for zz in range(znum):
                        zl = step * zz
                        zr = step * zz + cube_size
                        if zr > x.shape[3]:
                            zr = x.shape[3]
                            zl = x.shape[3] - cube_size
                        tmp.append([xl, xr, yl, yr, zl, zr])
            while (len(tmp) % self.batch_size) != 0:
                tmp.append(tmp[0])
            file_dic[f] = tmp

        file_list, pos_list = [], []
        for f in self.file_list:
            file_list += [f for i in range(len(file_dic[f]))]
            pos_list += file_dic[f]
        self.file_list = file_list
        return file_dic, pos_list

    def crop(self, data, pos):
        xl, xr, yl, yr, zl, zr = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
        data_crop = data[:, xl:xr, yl:yr, zl:zr]
        return data_crop

    def process_imgmsk(self, data):
        data = data.astype(np.float32)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048

        return data, data2

    def __getitem__(self, item):
        name = self.file_list[item]
        if name != self.last_name:
            img = sitk.ReadImage(self.root+'/data/'+name+ 'data_cut' + '.nii.gz')
            img = sitk.GetArrayFromImage(img)
            img=img-1024
            img, img2 = self.process_imgmsk(img)
            img = np.array([img, img2])
            self.img = img
            self.last_name = name
        else:
            img = self.img
        img_crop = self.crop(img, self.pos_list[item])
        pos = np.array(self.pos_list[item])

        return torch.from_numpy(img_crop.astype(np.float32)), name, torch.from_numpy(pos)

class SegTestCropData(Dataset):
    def __init__(self, file_path, data_root, batch_size, cube_size=128, step=64):
        self.file_list = load_json_file(file_path, folder='-1', mode=['test'])
        # self.file_list = load_json_file(file_path, folder='0', mode=['val'])
        self.root = data_root
        self.cube_size = cube_size
        self.batch_size = batch_size
        self.step = step
        self.file_dic, self.pos_list = self.crop_pos() 
        self.last_name = ''
        self.img = None

    def __len__(self):
        return len(self.file_list)

    def crop_pos(self):
        file_dic = {}
        cube_size, step = self.cube_size, self.step
        for f in self.file_list:
            tmp = []
            img = sitk.ReadImage(self.root+'/data/'+f+ 'data_cut' + '.nii.gz')
            x = sitk.GetArrayFromImage(img)
            x = x[np.newaxis, ...]
            xnum = (x.shape[1] - cube_size) // step + 1 if (x.shape[1] - cube_size) % step == 0 else \
                (x.shape[1] - cube_size) // step + 2
            ynum = (x.shape[2] - cube_size) // step + 1 if (x.shape[2] - cube_size) % step == 0 else \
                (x.shape[2] - cube_size) // step + 2
            znum = (x.shape[3] - cube_size) // step + 1 if (x.shape[3] - cube_size) % step == 0 else \
                (x.shape[3] - cube_size) // step + 2
            for xx in range(xnum):
                xl = step * xx
                xr = step * xx + cube_size
                if xr > x.shape[1]:
                    xr = x.shape[1]
                    xl = x.shape[1] - cube_size
                for yy in range(ynum):
                    yl = step * yy
                    yr = step * yy + cube_size
                    if yr > x.shape[2]:
                        yr = x.shape[2]
                        yl = x.shape[2] - cube_size
                    for zz in range(znum):
                        zl = step * zz
                        zr = step * zz + cube_size
                        if zr > x.shape[3]:
                            zr = x.shape[3]
                            zl = x.shape[3] - cube_size
                        tmp.append([xl, xr, yl, yr, zl, zr])
            while (len(tmp) % self.batch_size) != 0:
                tmp.append(tmp[0])
            file_dic[f] = tmp

        file_list, pos_list = [], []
        for f in self.file_list:
            file_list += [f for i in range(len(file_dic[f]))]
            pos_list += file_dic[f]
        self.file_list = file_list
        return file_dic, pos_list

    def crop(self, data, pos):
        xl, xr, yl, yr, zl, zr = pos[0], pos[1], pos[2], pos[3], pos[4], pos[5]
        data_crop = data[:, xl:xr, yl:yr, zl:zr]
        return data_crop

    def process_imgmsk(self, data):
        data = data.astype(np.float32)
        data2 = data.copy()
        data2[data2 > 500] = 500
        data2[data2 < -1000] = -1000
        data2 = (data2 + 1000) / 1500
        data[data > 1024] = 1024
        data[data < -1024] = -1024
        data = (data + 1024) / 2048

        return data, data2

    def __getitem__(self, item):
        name = self.file_list[item]
        if name != self.last_name:
            img = sitk.ReadImage(self.root+'/data/'+name+ 'data_cut' + '.nii.gz')
            img = sitk.GetArrayFromImage(img)
            img=img-1024
            img, img2 = self.process_imgmsk(img)
            img = np.array([img, img2])
            self.img = img
            self.last_name = name
        else:
            img = self.img
        img_crop = self.crop(img, self.pos_list[item])
        pos = np.array(self.pos_list[item])

        return torch.from_numpy(img_crop.astype(np.float32)), name, torch.from_numpy(pos)



if __name__ == '__main__':
    # # npy2niigz()
    # p1 = nibabel.load('./data/pred2/ATM_001_0000.nii.gz')
    # p1 = p1.get_fdata()
    # p2 = nibabel.load('./data/pred_1/ATM_001_0000.nii.gz')
    # p2 = p2.get_fdata()[0]
    # print(p1.shape, p2.shape, 2 * (p1 * p2).sum() / (p1 + p2).sum())
    print(1)
